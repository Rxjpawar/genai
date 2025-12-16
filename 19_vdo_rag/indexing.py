from pathlib import Path
from pytubefix import YouTube
from moviepy.video.io.VideoFileClip import VideoFileClip
from faster_whisper import WhisperModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


video_url = "https://youtu.be/sfyL4BswUeE"
video_dir = Path("video")
video_dir.mkdir(exist_ok=True)

print("Downloading the video")
yt = YouTube(video_url)
stream = yt.streams.get_highest_resolution()
video_path = stream.download(output_path=video_dir)

print(f"Downloaded the video : \n {video_path}")


video = VideoFileClip(video_path)
audio_path  = str(Path(video_path).with_suffix(".mp3"))
video.audio.write_audiofile(audio_path)

print(f"Extracted audio : {audio_path}")


print("Transcripting the audio")
model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe(audio_path)
transcript_text = " ".join(segment.text for segment in segments)
print("Transcription completed")


text_splitter  = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 200
)

document = text_splitter.split_documents([Document(page_content=transcript_text)])
print(f'Total chunks created {len(document)}')

print("Loading the embedding model")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)
print("Loaded the embdding model")

print("Storing the vector db")

vector_store=QdrantVectorStore.from_documents(
    documents = document,
    url = "http://localhost:6333",
    collection_name = "video_rag",
    embedding= embedding_model
)
print("Indexing is completed")