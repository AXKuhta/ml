
import bz2

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

#################################################################################################################
# HF setup
#

from huggingface_hub import logout, login, create_repo

login()

#################################################################################################################
# Datasets
#

from datasets import load_dataset

# Dataset 1 (Unused)
#dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

#def get_training_corpus():
#    for i in range(0, len(dataset), 1000):
#        yield dataset[i : i + 1000]["text"]

# Dataset 2
def stream_strings():
	with bz2.BZ2File("bio-recipe-datasettxt.bz2") as f:
		words = []
		types = []
		for line in f:
			if line == b"\n":
				yield dict(
					words=words,
					labels=types
				)
				words = []
				types = []
				continue

			assert b"\t" in line, line

			word, type = line.decode().strip().split("\t")
			words.append(word)
			types.append(type)

z = list(stream_strings())

# Training tokenizer on recipes will lead to minorly worse scores but shorter tokenizations vs wikitext and stock tokenizer
def get_training_corpus():
        for example in z:
                yield example["words"]

#################################################################################################################
# Tokenizer training
#

from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer


from transformers import TokenizersBackend

#
# class example: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py
#
class CustomTokenizer(TokenizersBackend):
  vocab_files_names = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}
  model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
  model = models.WordPiece
  padding_side = "left"

  def __init__(self, vocab=None, merges=None,
               cls_token="[CLS]",
               pad_token="[PAD]",
               unk_token="[UNK]",
               sep_token="[SEP]",
               mask_token="[MASK]",
               **kwargs):
    if vocab is None:
      vocab = {
          str(pad_token): 0,
          str(unk_token): 100,
          str(cls_token): 101,
          str(sep_token): 102,
          str(mask_token): 103,
      }
    self._vocab = vocab
    self._merges = merges
    self._tokenizer = Tokenizer(models.WordPiece(
        vocab=self._vocab,
        merges=self._merges,
        unk_token="[UNK]")
    )
    self._tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    self._tokenizer.normalizer = normalizers.Sequence(
      [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    cls_token_id = self._tokenizer.token_to_id("[CLS]")
    sep_token_id = self._tokenizer.token_to_id("[SEP]")
    self._tokenizer.post_processor = processors.TemplateProcessing(
      single=f"[CLS]:0 $A:0 [SEP]:0",
      pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
      special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )

    super().__init__(
        vocab=vocab,
        merges=merges,
        cls_token="[CLS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        **kwargs
    )

tokenizer = CustomTokenizer()

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
tokenizer._tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)


#################################################################################################################
# Model training
#

le = LabelEncoder()
le.fit([v for x in z for v in x["labels"]])

id2label = dict(enumerate(x.item() for x in le.classes_))
label2id = {v:k for k,v in id2label.items()}

def tokenize_and_align_labels(example):
	tokenized_inputs = tokenizer(
		example.get("words"), truncation=True, is_split_into_words=True
	)
	world_labels = le.transform(example.get("labels"))
	previous_wid = None
	labels = []

	for wid in tokenized_inputs.word_ids():
		if wid is None:
			labels.append(-100)
		elif wid != previous_wid:
			labels.append(world_labels[wid].item())
		else:
			labels.append(-100)

		previous_wid = wid

	tokenized_inputs["labels"] = labels
	return tokenized_inputs

ds = Dataset.from_list(z)

tokenized_ds = ds.map(tokenize_and_align_labels)

train, tmp = tokenized_ds.train_test_split(0.04, seed=42).values()
val, test = tmp.train_test_split(0.5, seed=42).values()

ds = DatasetDict(dict(
	train=train,
	val=val,
	test=test
))

ds = ds.remove_columns("words")

import numpy as np

def compute_metrics(eval_preds):
	logits, labels = eval_preds
	predictions = np.argmax(logits, axis=-1)

	# Remove ignored index (special tokens) and convert to labels
	true_labels = [le.inverse_transform([l for l in label if l != -100]) for label in labels]
	preds = [
		le.inverse_transform([p for (p, l) in zip(prediction, label) if l != -100])
		for prediction, label in zip(predictions, labels)
	]

	report = classification_report(
		[x for y in true_labels for x in y],
		[x for y in preds for x in y]
	)

	print(report)

	return dict(report=report)



from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
	"bert-base-uncased",
	id2label=id2label,
	label2id=label2id,
)

from transformers import TrainingArguments

args = TrainingArguments(
	"bert-finetuned-ner",
	save_strategy="epoch",
	learning_rate=4e-5, # 2x
	num_train_epochs=6, # Training extension
	weight_decay=0.01,
	push_to_hub=True,
        hub_model_id="AXKuhta/bert-finetuned-ner",
	eval_strategy="steps",
	eval_steps=500,
	per_device_train_batch_size=16, #
	per_device_eval_batch_size=16   # same result but with less steps (faster)
)

from transformers import Trainer

trainer = Trainer(
	model=model,
	args=args,
	train_dataset=ds["train"],
	eval_dataset=ds["val"],
	data_collator=data_collator,
	compute_metrics=compute_metrics
)
trainer.evaluate()
trainer.train()

model.save_pretrained("ner-fin")
tokenizer.save_pretrained("ner-fin")

print("Eval on test:")
_ = trainer.evaluate(test)

trainer.push_to_hub("End of training")

model.config.push_to_hub("AXKuhta/bert-finetuned-ner")
tokenizer.push_to_hub("AXKuhta/bert-finetuned-ner")

