from typing import *
import random

random.seed(42)

from datasets import Dataset
from transformers import Pipeline, pipeline
from transformers.pipelines import LLMProxyPipeline
import transformers

from .base import Evaluator
from .utils import DatasetColumn
from ..utils.logging import get_logger

logger = get_logger(__name__)


class LLMProxyEvaluator(Evaluator):
    def __init__(self, task="llm-proxy", default_metric_name=None):
        super().__init__(task, default_metric_name=default_metric_name)

    def prepare_data(
        self, data: Dataset, input_variables: Sequence[Text], label_column: str
    ):
        """Prepare data."""
        if data is None:
            raise ValueError(
                "Please specify a valid `data` object - either a `str` with a name or a `Dataset` object."
            )
        self.check_required_columns(
            data,
            {
                **{
                    input_variable: input_variable for input_variable in input_variables
                },
                "label_column": label_column,
            },
        )
        return {"references": data[label_column]}, {
            "inputs": {
                input_variable: DatasetColumn(data, input_variable)
                for input_variable in input_variables
            }
        }

    def predictions_processor(self, llm_outputs, process_predictions_fn, *_):
        return process_predictions_fn(llm_outputs)

    def compute(
        self,
        model_or_pipeline: "Pipeline" = None,
        data: Union[str, Dataset] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        metric: Union[str, "EvaluationModule"] = None,
        tokenizer: Optional[Union[str, "PreTrainedTokenizer"]] = None,  # noqa: F821
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        device: int = None,
        random_state: Optional[int] = None,
        label_mapping: Optional[Dict[str, "Number"]] = None,
        input_variables: Optional[List[str]] = None,
        label_column: str = "label",
        return_predictions: bool = False,
        predictions_processor_fn: Optional[Callable] = None,
    ) -> Tuple[Dict[str, float], Any]:
        metric_result = {}
        self.check_for_mismatch_in_device_setup(device, model_or_pipeline)

        data = self.load_data(data=data, subset=subset, split=split)
        metric_inputs, pipe_inputs = self.prepare_data(
            data=data, input_variables=input_variables, label_column=label_column
        )
        pipe = self.prepare_pipeline(model_or_pipeline=model_or_pipeline)

        metric = self.prepare_metric(metric)

        # Compute predictions
        print(f"Invoking pipeline")
        predictions, perf_results = self.call_pipeline(pipe, **pipe_inputs)
        print(f"Pipeline invoked")
        predictions = self.predictions_processor(predictions, predictions_processor_fn)
        metric_inputs.update({"predictions": predictions["predictions"]})

        # Compute metrics from references and predictions
        metric_output = self.compute_metric(
            metric=metric,
            metric_inputs=metric_inputs,
            strategy=strategy,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
        )

        metric_result.update(metric_output)
        metric_result.update(perf_results)

        return metric_result if not return_predictions else (metric_result, predictions)

    def prepare_pipeline(
        self,
        model_or_pipeline: LLMProxyPipeline,
    ):
        if (
            isinstance(model_or_pipeline, str)
            or isinstance(model_or_pipeline, transformers.PreTrainedModel)
            or isinstance(model_or_pipeline, transformers.TFPreTrainedModel)
        ):
            pipe = pipeline(self.task, model=model_or_pipeline, return_predictions=True)
        else:
            if model_or_pipeline is None:
                pipe = pipeline(self.task, return_predictions=True)
            else:
                pipe = model_or_pipeline
        return pipe
