import numpy as np
import pandas as pd
import re
from typing import List

class TFIDF:
    def __init__(self):
        # Từ điển lưu trữ IDF
        self.idf_dict = {}
        # Từ vựng
        self.vocabulary = []
    
    def remove_tags(self, text: str) -> np.ndarray:
        """
        Tiền xử lý văn bản
        
        Parameters:
        text (str): Văn bản đầu vào
        
        Returns:
        numpy.ndarray: Mảng các từ đã xử lý
        """
        result = re.sub(r'<.*?>','',text)           #xoá HTML tags, tiền tố r nghĩa là raw
        result = re.sub('https://.*','',result)     #xoá URLs
        result = re.sub(r'[^\w\s]', '',result)      #xoá non-alphanumeric characters ví dụ như các dấu câu
        result = result.lower()                     # Chuyển về chữ thường
        
        # Tách từ và chuyển thành numpy array
        return np.array(result.split())
    
    def fit(self, documents: List[str]) -> 'TFIDF':
        """
        Học từ vựng và tính IDF
        
        Parameters:
        documents (List[str]): Danh sách các văn bản
        
        Returns:
        self: Đối tượng hiện tại
        """
        # Xử lý văn bản thành numpy arrays
        processed_docs = [self.remove_tags(doc) for doc in documents]
        
        # Chuyển sang DataFrame để dễ xử lý
        df_docs = pd.DataFrame({'words': processed_docs})
        # Tổng số văn bản
        total_docs = len(documents)
        
        # Đếm số văn bản chứa từng từ
        word_doc_count = {}
        for doc_words in processed_docs:
            unique_words = set(doc_words)  # Chỉ đếm mỗi từ một lần trong một văn bản
            for word in unique_words:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1
        
        # Tính IDF sử dụng numpy
        self.idf_dict = {
            word: np.log(total_docs / (count )) + 1 
            for word, count in word_doc_count.items()
        }
        
        
        # Lưu từ vựng
        self.vocabulary = list(word_doc_count.keys())
        
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Chuyển đổi văn bản thành ma trận TF-IDF
        
        Parameters:
        documents (List[str]): Danh sách các văn bản
        
        Returns:
        numpy.ndarray: Ma trận TF-IDF
        """
        # Khởi tạo ma trận kết quả
        tfidf_matrix = np.zeros((len(documents), len(self.vocabulary)))
        
        # Xử lý từng văn bản
        for i, doc in enumerate(documents):
            # Tiền xử lý văn bản
            words = self.remove_tags(doc)
            
            # Tính Term Frequency (TF) sử dụng pandas
            word_counts = pd.Series(words).value_counts()
            total_words = len(words)
            
            # Tạo vector cho văn bản
            for j, vocab_word in enumerate(self.vocabulary):
                # Tính TF
                tf = word_counts.get(vocab_word, 0) / total_words
                
                # Tính TF-IDF
                idf = self.idf_dict.get(vocab_word, 0)
                tfidf_matrix[i, j] = tf * idf
        
        return tfidf_matrix

# Ví dụ sử dụng
def main():
    # Tập văn bản mẫu
    documents = [
        "one of the other reviewers has mentioned that after watching just 1 oz episode youll be hooked they are right as this is exactly what happened with methe first thing that struck me about oz was its brutality and unflinching scenes of violence which set in right from the word go trust me this is not a show for the faint hearted or timid this show pulls no punches with regards to drugs sex or violence its is hardcore in the classic use of the wordit is called oz as that is the nickname given to the oswald maximum security state penitentary it focuses mainly on emerald city an experimental section of the prison where all the cells have glass fronts and face inwards so privacy is not high on the agenda em city is home to manyaryans muslims gangstas latinos christians italians irish and moreso scuffles death stares dodgy dealings and shady agreements are never far awayi would say the main appeal of the show is due to the fact that it goes where other shows wouldnt dare forget pretty pictures painted for mainstream audiences forget charm forget romanceoz doesnt mess around the first episode i ever saw struck me as so nasty it was surreal i couldnt say i was ready for it but as i watched more i developed a taste for oz and got accustomed to the high levels of graphic violence not just violence but injustice crooked guards wholl be sold out for a nickel inmates wholl kill on order and get away with it well mannered middle class inmates being turned into prison bitches due to their lack of street skills or prison experience watching oz you may become comfortable with what is uncomfortable viewingthats if you can get in touch with your darker side",
        "a wonderful little production the filming technique is very unassuming very oldtimebbc fashion and gives a comforting and sometimes discomforting sense of realism to the entire piece the actors are extremely well chosen michael sheen not only has got all the polari but he has all the voices down pat too you can truly see the seamless editing guided by the references to williams diary entries not only is it well worth the watching but it is a terrificly written and performed piece a masterful production about one of the great masters of comedy and his life the realism really comes home with the little things the fantasy of the guard which rather than use the traditional dream techniques remains solid then disappears it plays on our knowledge and our senses particularly with the scenes concerning orton and halliwell and the sets particularly of their flat with halliwells murals decorating every surface are terribly well done",
        "i thought this was a wonderful way to spend time on a too hot summer weekend sitting in the air conditioned theater and watching a lighthearted comedy the plot is simplistic but the dialogue is witty and the characters are likable even the well bread suspected serial killer while some may be disappointed when they realize this is not match point 2 risk addiction i thought it was proof that woody allen is still fully in control of the style many of us have grown to lovethis was the most id laughed at one of woodys comedies in years dare i say a decade while ive never been impressed with scarlet johanson in this she managed to tone down her sexy image and jumped right into a average but spirited young womanthis may not be the crown jewel of his career but it was wittier than devil wears prada and more interesting than superman a great comedy to go see with friends"
    ]
    
    # Khởi tạo đối tượng TF-IDF
    vectorizer = TFIDF()
    
    # Học từ vựng và IDF
    vectorizer.fit(documents)

    # # In ra từ vựng
    # print("Từ vựng:")
    # print(vectorizer.vocabulary)
    
    # In ra IDF của các từ
    print("\nIDF của các từ:")
    for word, idf in vectorizer.idf_dict.items():
        print(f"{word}: {idf:.4f}")
    
    # Chuyển đổi văn bản thành ma trận TF-IDF
    tfidf_matrix = vectorizer.transform(documents)
    
    # Tạo DataFrame để hiển thị kết quả
    tfidf_df = pd.DataFrame(
        tfidf_matrix, 
        columns=vectorizer.vocabulary
    )
    
    print("\nMa trận TF-IDF:")
    print(tfidf_df)

if __name__ == "__main__":
    main()