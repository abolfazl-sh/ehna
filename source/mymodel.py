import csv
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses
from sentence_transformers import evaluation
import torch
print(torch.cuda.is_available())

model_id = "intfloat/multilingual-e5-large"
model = SentenceTransformer(model_id, device='cuda')
qustions_answers = []
with open("./data/questions_and_replies.csv", encoding="utf-8") as csvfile:

    data = csv.reader(csvfile, delimiter=",")
    for d in data:
        qustions_answers.append(InputExample(texts=[d[0], d[1]]))


loader = DataLoader(qustions_answers, batch_size=10)
loss = losses.MultipleNegativesRankingLoss(model)

EPOCHS = 2
warmup_steps = int(len(loader) * EPOCHS * 0.1)
model.fit(
    train_objectives=[(loader, loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path='finetune',
    show_progress_bar=True,
    #evaluator=evaluator,
    evaluation_steps=50,
)
"""evaluator = evaluation.InformationRetrievalEvaluator(
    queries={idx:d.texts[0] for idx, d in enumerate(qustions_answers)},
    corpus={idx:d.texts[1] for idx, d in enumerate(qustions_answers)},
    relevant_docs={idx : idx for idx, _ in enumerate(qustions_answers)})"""
