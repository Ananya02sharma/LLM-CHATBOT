import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

class WebScraper:
    """
    A class to fetch and extract content from a given URL.
    """

    def __init__(self, url):
        """
        Initialize the WebScraper with the URL.
        
        :param url: The URL of the website to scrape.
        """
        self.url = url

    def fetch_content(self):
        """
        Fetches content from the URL and extracts information.
        
        :return: A dictionary containing extracted data or None if an error occurs.
        """
        try:
            response = requests.get(self.url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            soup = BeautifulSoup(response.content, 'html.parser')
            return self.extract_information(soup)
        except requests.RequestException as e:
            print(f"Error fetching the website: {e}")
            return None

    def extract_information(self, soup):
        """
        Extracts information from the BeautifulSoup object.
        
        :param soup: A BeautifulSoup object containing the parsed HTML.
        :return: A dictionary with extracted text data.
        """
        data = {
            "title": soup.title.string if soup.title else "No title found",
            "headings": {
                "h1": [h1.get_text() for h1 in soup.find_all('h1')],
                "h2": [h2.get_text() for h2 in soup.find_all('h2')],
                "h3": [h3.get_text() for h3 in soup.find_all('h3')],
                "h4": [h4.get_text() for h4 in soup.find_all('h4')],
                "h5": [h5.get_text() for h5 in soup.find_all('h5')],
                "h6": [h6.get_text() for h6 in soup.find_all('h6')],
            },
            "paragraphs": [p.get_text() for p in soup.find_all('p')],
            "span": [p.get_text() for p in soup.find_all('div')],
            "lists": {
                "ul": [[li.get_text() for li in ul.find_all('li')] for ul in soup.find_all('ul')],
                "ol": [[li.get_text() for li in ol.find_all('li')] for ol in soup.find_all('ol')]
            },
        }
        return data

class TextEmbedder:
    """
    A class to handle text embedding using a pre-trained model.
    """

    def __init__(self):
        """
        Initialize the TextEmbedder with the SentenceTransformer model.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_text(self, texts):
        """
        Converts a list of texts into vector embeddings.
        
        :param texts: List of text strings to be embedded.
        :return: A numpy array of text embeddings or None if an error occurs.
        """
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            print(f"Error embedding text: {e}")
            return None

class VectorIndex:
    """
    A class to create, save, load, and search a FAISS index of vectors.
    """

    def __init__(self):
        """
        Initialize the VectorIndex with an empty FAISS index.
        """
        self.index = None

    def create_index(self, vectors):
        """
        Creates a FAISS index with the given vectors.
        
        :param vectors: A numpy array of vectors to be indexed.
        """
        try:
            dimension = vectors.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(vectors)
        except Exception as e:
            print(f"Error creating FAISS index: {e}")

    def save_index(self, filename="index.index"):
        """
        Saves the FAISS index to a file.
        
        :param filename: The filename where the index will be saved.
        """
        try:
            faiss.write_index(self.index, filename)
        except Exception as e:
            print(f"Error saving FAISS index: {e}")

    def load_index(self, filename="index.index"):
        """
        Loads a FAISS index from a file.
        
        :param filename: The filename from which the index will be loaded.
        """
        try:
            self.index = faiss.read_index(filename)
        except Exception as e:
            print(f"Error loading FAISS index: {e}")

    def search(self, query_vector, k=5):
        """
        Searches the FAISS index for the nearest neighbors of the query vector.
        
        :param query_vector: The vector representation of the query.
        :param k: The number of nearest neighbors to retrieve.
        :return: Distances and indices of the nearest neighbors or None if an error occurs.
        """
        try:
            D, I = self.index.search(np.array([query_vector]), k)
            return D, I
        except Exception as e:
            print(f"Error searching FAISS index: {e}")
            return None, None

class ChatBot:
    """
    A class to interact with the Groq chatbot API.
    """

    def __init__(self, api_key):
        """
        Initialize the ChatBot with the Groq API key.
        
        :param api_key: The API key for the Groq service.
        """
        self.client = Groq(api_key=api_key)

    def get_response(self, query, context):
        """
        Gets a response from the Groq API based on the query and context.
        
        :param query: The user's query.
        :param context: The context to be used for generating the response.
        :return: The chatbot's response or None if an error occurs.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": f"Summarize the given context as the answer to the given query: {query}\nContext: {context}"},
                ],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error interacting with chatbot API: {e}")
            return None

def main(query):
    """
    Main function to execute the chatbot workflow.
    
    :param query: The user's query to be processed.
    :return: The response from the chatbot.
    """
    url = "https://en.wikipedia.org/wiki/Politics_of_India"
    
    # Initialize the WebScraper and fetch content
    scraper = WebScraper(url)
    extracted_data = scraper.fetch_content()
    
    if extracted_data:
        # Prepare text for embedding
        texts = [extracted_data['title']] + extracted_data['paragraphs']
        
        # Initialize the TextEmbedder and embed the text
        embedder = TextEmbedder()
        vectors = embedder.embed_text(texts)
        
        if vectors is not None:
            # Initialize VectorIndex, create, and save the index
            vector_index = VectorIndex()
            vector_index.create_index(vectors)
            vector_index.save_index()

            # Load index and search for relevant content
            vector_index.load_index()
            query_vector = embedder.embed_text([query])[0]  # Embed the query
            _, indices = vector_index.search(query_vector, k=2)

            if indices is not None:
                # Retrieve context for the query
                context = " ".join(texts[i] for i in indices[0])
                
                # Initialize ChatBot and get response
                chatbot = ChatBot(api_key='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')
                response = chatbot.get_response(query, context)
                return response

    return "No response generated."

if __name__ == "__main__":
    print("Hello chatbot -----")
    query = input("Ask Query ---- ")
    response = main(query)
    print("Answer ----", response)
