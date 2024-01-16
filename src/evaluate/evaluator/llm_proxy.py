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
        return {"references": data[label_column]}, {"inputs": {
            input_variable: DatasetColumn(data, input_variable)
            for input_variable in input_variables
        }}

    def predictions_processor(self, predictions, label_mapping):
        def get_fallback_score(s):
            """As a fallback, return any digit present in the string.
            If even that fails, return -1.
            """
            import re
            numbers = re.findall(r'-?\d+\.?\d*', s)
            if len(numbers) == 0:
                return -1
            return numbers[0]
        processed_predictions = []
        for element in predictions:
            try:
                processed_predictions.append(float(element))
            except ValueError:
                processed_predictions.append(get_fallback_score(element))
        return {"predictions": processed_predictions}

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
    ) -> Tuple[Dict[str, float], Any]:
        result = {}
        self.check_for_mismatch_in_device_setup(device, model_or_pipeline)

        data = self.load_data(data=data, subset=subset, split=split)
        metric_inputs, pipe_inputs = self.prepare_data(
            data=data, input_variables=input_variables, label_column=label_column
        )
        pipe = self.prepare_pipeline(
            model_or_pipeline=model_or_pipeline
        )

        metric = self.prepare_metric(metric)

        # Compute predictions
        predictions, perf_results = self.call_pipeline(pipe, **pipe_inputs)
        predictions = self.predictions_processor(
            predictions, label_mapping=label_mapping
        )
        metric_inputs.update(predictions)

        # Compute metrics from references and predictions
        metric_results = self.compute_metric(
            metric=metric,
            metric_inputs=metric_inputs,
            strategy=strategy,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
        )

        result.update(metric_results)
        result.update(perf_results)

        return result if not return_predictions else (result, predictions)

    def prepare_pipeline(
        self,
        model_or_pipeline: LLMProxyPipeline,
    ):
        if (
            isinstance(model_or_pipeline, str)
            or isinstance(model_or_pipeline, transformers.PreTrainedModel)
            or isinstance(model_or_pipeline, transformers.TFPreTrainedModel)
        ):
            pipe = pipeline(
                self.task,
                model=model_or_pipeline,
                return_predictions=True
            )
        else:
            if model_or_pipeline is None:
                pipe = pipeline(self.task, return_predictions=True)
            else:
                pipe = model_or_pipeline
        return pipe
