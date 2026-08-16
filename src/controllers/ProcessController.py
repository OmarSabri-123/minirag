import os
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import ImageRefMode
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from transformers import AutoTokenizer
from functools import lru_cache
from models import ResponseEnumeration, ProcessingEnums
from .BaseController import BaseController
from .ProjectController import ProjectController
from logger import logger

@lru_cache(maxsize=2)
def get_chunk_tokenizer(tokenizer_name: str):
    # cached so the tokenizer is loaded once, not on every request
    logger.info(f"Loading chunking tokenizer: {tokenizer_name}")
    return AutoTokenizer.from_pretrained(tokenizer_name)

class ProcessController(BaseController):
    def __init__(self,project_id: str):
        super().__init__()
        self.project_id = project_id
        self.header_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        # ordered from the strongest boundary to the weakest, so the splitter
        # only falls back to a weaker one when a chunk is still too large
        self.text_separators = [
            "\n\n", "\n",                       # paragraph / line
            ".", "!", "?",                      # latin / cyrillic sentence end
            "。", "！", "？",       # 。！？ cjk sentence end
            "۔", "؟",                 # ۔ ؟ arabic / urdu sentence end
            "।", "॥",                 # । ॥ devanagari danda
            "።",                           # ። ethiopic sentence end
            ";", ":", ",",                      # latin clause
            "؛", "،",                 # ؛ ، arabic clause
            "、", "，",                 # 、，cjk clause
            " ", "　", "​",            # space, ideographic space, zero-width space (thai/khmer/burmese)
            ""                                  # last resort: split anywhere
        ]

    def get_process_path(self):
        process_path = ProjectController().get_project_path(self.project_id)
        return process_path
    
    def get_extension(self, file_name: str) -> str:
        return os.path.splitext(file_name)[-1].lower()
    
    def check_process_file(self, file_name: str):

        file_path = os.path.join(self.get_process_path(), file_name)
        if not os.path.exists(file_path):
            return False, ResponseEnumeration.FILE_NOT_EXIST.value.format(file_path=file_path)
        
        return True, ResponseEnumeration.FILE_EXIST.value.format(file_path=file_path)
    
    def convert_process_file_into_markdown(self, file_name: str):

        pipline_options = PdfPipelineOptions()
        pipline_options.do_formula_enrichment = True
        pipline_options.do_ocr = True
        pipline_options.generate_picture_images = True
        pipline_options.images_scale =3
        
        converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipline_options)
    })
        file_path = os.path.join(self.get_process_path(), file_name)

        result = converter.convert(file_path)

        return result
    
    def export_function_md_with_image_ref(self,conv_res,replace_blank:str="_"):

        doc_filename = conv_res.input.file.stem.replace(" ", replace_blank)
        # Save markdown with externally referenced pictures
        md_filename = os.path.join(self.get_process_path(), f"{doc_filename}-with-image-refs.md")
        conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED, include_annotations=True)
        file_id = f"{doc_filename}-with-image-refs.md"
        return file_id
    
    def chunk_text_markdown(self, text: str):

        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.header_to_split_on, strip_headers=False)
        chunks = splitter.split_text(text)
        return chunks
    
    def recursive_chunk_text(self, documents):

        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            get_chunk_tokenizer(self.app_settings.CHUNK_TOKENIZER_NAME),
            chunk_size=self.app_settings.CHUNK_SIZE_TOKENS,
            chunk_overlap=self.app_settings.CHUNK_OVERLAP_TOKENS,
            separators=self.text_separators,
            keep_separator="end"
        )
        chunks = splitter.split_documents(documents)
        return chunks
    
    def load_markdown_or_txt_file(self, file_name:str):

        file_path = os.path.join(self.get_process_path(), file_name)

        loader = TextLoader(file_path,encoding="utf-8")
        documents = loader.load()
        return documents
    
    def load_docx_file(self, file_name:str):

        file_path = os.path.join(self.get_process_path(), file_name)

        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        return documents

    def parse_csv_file(self, file_name:str):
        # Placeholder for CSV parsing logic
        # You can use pandas or csv module to read and process CSV files
        file_path = os.path.join(self.get_process_path(), file_name)
        df = pd.read_csv(file_path)

        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)

        documents = []
        for i, row in df.iterrows():
            content = ", ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])

            documents.append(
                Document(page_content=content, metadata={"source": file_name, "row_index": i})
                )
            
        return documents
            
    def load_and_export(self, file_name:str):

        ext = self.get_extension(file_name)

        try:
            if ext == ProcessingEnums.PDF.value:
                conv_res = self.convert_process_file_into_markdown(file_name)
                file_id = self.export_function_md_with_image_ref(conv_res)
                documents = self.load_markdown_or_txt_file(file_id)
                chunks = self.chunk_text_markdown(documents[0].page_content)
                logger.info(f"Initial chunks from PDF conversion: {len(chunks)}")
            
            elif ext == ProcessingEnums.MARKDOWN.value:
                documents = self.load_markdown_or_txt_file(file_name)
                chunks = self.chunk_text_markdown(documents[0].page_content)
                chunks = self.recursive_chunk_text(chunks)
                logger.info(f"Initial chunks from Markdown file: {len(chunks)}")
            
            elif ext == ProcessingEnums.TXT.value:
                documents = self.load_markdown_or_txt_file(file_name)
                chunks = self.recursive_chunk_text(documents)
                logger.info(f"Initial chunks from TXT file: {len(chunks)}")

            elif ext == ProcessingEnums.DOCX.value:
                documents = self.load_docx_file(file_name)
                chunks = self.recursive_chunk_text(documents)
                logger.info(f"Initial chunks from DOCX file: {len(chunks)}")

            elif ext == ProcessingEnums.CSV.value:
                chunks = self.parse_csv_file(file_name)
                logger.info(f"Initial chunks from CSV file: {len(chunks)}")

            return chunks
        
        except Exception as e:
            logger.error(f"Error in load_and_export: {str(e)}")
            raise e
    


