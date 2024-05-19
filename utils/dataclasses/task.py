from typing import Union, Dict, List, Tuple

from config.constants import TEXT_PROMPT, LABEL_PROMPT


class Task:
    def __init__(self, dataset_name: str, dataset_train: str, dataset_val: str, dataset_test: str,
                 dataset_column_text: str, dataset_column_label: str, bert_fine_tuned_model: str,
                 roberta_fine_tuned_model: str, distilbert_fine_tuned_model: str, roberta_base_model: str,
                 distilbert_base_model: str, bert_base_model: str, llama_model: str,
                 labels_str_int_maps: Union[Dict, None], default_lr: float, llama_lr: float,
                 test_sample: Union[int, None], train_sample: Union[int, None], hp_search_test_sample: Union[int, None],
                 hp_search_train_sample: Union[int, None], name: str, hp_search_n_trials: int, llama_task_prompt: str,
                 llama_few_shots_prompt: List[Tuple[str, int]], llama_is_for_seq_class = False,
                 is_llama_set_max_len = False, llama_explained_tokenizer_max_length: int = -1,
                 llama_explainer_tokenizer_max_length: int = -1):
        self.dataset_name = dataset_name
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.dataset_column_text = dataset_column_text
        self.dataset_column_label = dataset_column_label
        self.bert_fine_tuned_model = bert_fine_tuned_model
        self.roberta_fine_tuned_model = roberta_fine_tuned_model
        self.distilbert_fine_tuned_model = distilbert_fine_tuned_model

        self.roberta_base_model = roberta_base_model
        self.distilbert_base_model = distilbert_base_model
        self.bert_base_model = bert_base_model

        self.llama_model = llama_model
        self.labels_str_int_maps = labels_str_int_maps
        self.labels_int_str_maps = {value: key for key, value in
                                    labels_str_int_maps.items()} if labels_str_int_maps else None
        self.default_lr = default_lr
        self.llama_lr = llama_lr
        self.test_sample = test_sample
        self.train_sample = train_sample
        self.hp_search_test_sample = hp_search_test_sample
        self.hp_search_train_sample = hp_search_train_sample
        self.hp_search_n_trials = hp_search_n_trials
        self.name = name
        self.llama_is_for_seq_class = llama_is_for_seq_class
        self.is_llama_set_max_len = is_llama_set_max_len
        self.llama_explained_tokenizer_max_length = llama_explained_tokenizer_max_length
        self.llama_explainer_tokenizer_max_length = llama_explainer_tokenizer_max_length
        self.llama_task_prompt = llama_task_prompt
        self.llama_few_shots = llama_few_shots_prompt  # for test only
        self.llama_few_shots_prompt = "\n\n".join(
            ["\n".join([TEXT_PROMPT + i[0], LABEL_PROMPT + str(i[1])]) for i in llama_few_shots_prompt])
