import openai
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# get envrioment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Q&Aのリストを用意
qa_list = [
    {"question": "Pythonで簡単にWebアプリを作るにはどうすればいいですか？", "answer": "PythonのWebフレームワークであるFlaskやDjangoがあります。"},
    {"question": "JavaScriptで画像を表示するにはどうすればいいですか？", "answer": "HTMLの<img>タグを使って画像を表示することができます。"},
    {"question": "MySQLで複数のテーブルを結合するにはどうすればいいですか？", "answer": "JOIN文を使ってテーブルを結合することができます。"},
]

def save_embeddings(qa_list, file_name="embeddings.pickle"):
    with open(file_name, "wb") as f:
        pickle.dump(qa_list, f)

def load_embeddings(file_name="embeddings.pickle"):
    try:
        with open(file_name, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Embeddingする関数を定義
def get_embedding(text):
    response = openai.Embedding.create(
      input=text,
      model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def create_embeddings_if_needed(qa_list, file_name="embeddings.pickle"):
    loaded_embeddings = load_embeddings(file_name)
    if loaded_embeddings is None:
        for qa in qa_list:
            qa["embedding"] = get_embedding(qa["question"])
        save_embeddings(qa_list, file_name)
    else:
        for i, qa in enumerate(qa_list):
            qa["embedding"] = loaded_embeddings[i]["embedding"]

# EmbeddingしたQ&Aのリストを作成
create_embeddings_if_needed(qa_list)

# 質問に対して関係ありそうな回答を選ぶ
# TODO:　閾値以下の質問・回答を全て取り込んでGPT3で回答すると良さそう
def chatbot(question):
    # 質問のEmbeddingを取得
    question_embedding = get_embedding(question)
    # 最も関連性が高いQ&Aを取得
    best_qa = max(qa_list, key=lambda x: cosine_similarity([x["embedding"]], [question_embedding])[0][0])
    # 質問と最も関連性が高いQ&AのEmbeddingを比較
    similarity = cosine_similarity([question_embedding], [best_qa["embedding"]])[0][0]
    # 類似度が一定値以上であれば、最も関連性が高いQ&Aの回答を返す
    if similarity > 0.5:
        return best_qa["answer"]
    else:
        return "回答が見つかりませんでした。"

# chatbotの動作テスト
question = "JavaScriptで画像を表示する方法"
answer = chatbot(question)
print(answer)
