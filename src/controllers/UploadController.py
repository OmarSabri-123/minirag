import os
from .BaseController import BaseController
from .ProjectController import ProjectController
from fastapi import UploadFile
from models import ResponseEnumeration, ProcessingEnums
import aiofiles
import asyncio
import docx2txt
import hashlib
import re

class UploadController(BaseController):
    def __init__(self):
        super().__init__()
        self.size_scale = 1024 * 1024  # Convert MB to bytes
        self.supported_file_types = [ext.value for ext in ProcessingEnums]
        # file types that can be read as text straight after upload, the rest
        # need extraction first and are detected during processing
        self.plain_text_types = [
            ProcessingEnums.TXT.value,
            ProcessingEnums.MARKDOWN.value,
            ProcessingEnums.CSV.value,
        ]
        self.language_sample_size = 4096  # characters used for detection
        self.language_mixed_threshold = 0.2  # share of letters that makes it bilingual
           # Inherited from BaseController


    def validate_file(self, file: UploadFile):
        # Validate file type
        if file.content_type not in self.app_settings.FILE_ALLOWED_EXTENSIONS:
            return False, ResponseEnumeration.INVALID_FILE_TYPE.value.format(file_type=file.content_type)
        
        # Validate file size
        if file.size is not None and file.size > self.app_settings.MAX_FILE_SIZE * self.size_scale:
            return False, ResponseEnumeration.FILE_TOO_LARGE.value.format(file_size=self.app_settings.MAX_FILE_SIZE)
        
        return True, ResponseEnumeration.FILE_VALIDATION_SUCCESS.value
    
    def generate_unique_filename(self, original_filename: str, project_id: str):
        
        random_key = self.generate_random_string()
        project_path = ProjectController().get_project_path(project_id)

        cleaned_file_name = self._get_clean_file_name(original_filename)

        new_file_location = os.path.join(project_path, f"{random_key}_{cleaned_file_name}")

        while os.path.exists(new_file_location):
            random_key = self.generate_random_string()
            new_file_location = os.path.join(project_path, f"{random_key}_{cleaned_file_name}")
        
        return new_file_location, f"{random_key}_{cleaned_file_name}"
    
    def get_file_type(self, original_filename: str):

        if not original_filename:
            return None, ResponseEnumeration.INVALID_FILE_TYPE.value.format(file_type=original_filename)

        ext = os.path.splitext(original_filename)[-1].lower()

        if ext not in self.supported_file_types:
            return None, ResponseEnumeration.INVALID_FILE_TYPE.value.format(file_type=ext)

        return ext, ResponseEnumeration.FILE_VALIDATION_SUCCESS.value

    async def write_file_and_get_hash(self, file: UploadFile, file_location: str):
        # stream the upload to disk and hash the same chunks, so the file is
        # only read once
        hasher = hashlib.sha256()

        async with aiofiles.open(file_location, "wb") as f:
            while content := await file.read(self.app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(content)
                hasher.update(content)

        return hasher.hexdigest()

    async def get_content_hash(self, file_location: str):
        # for files already saved on disk
        hasher = hashlib.sha256()

        async with aiofiles.open(file_location, "rb") as f:
            while content := await f.read(self.app_settings.FILE_DEFAULT_CHUNK_SIZE):
                hasher.update(content)

        return hasher.hexdigest()


    def detect_language(self, text: str):
        # arabic script vs latin script, counted over letters only so digits,
        # punctuation and markdown syntax do not skew the result
        if not text:
            return self.app_settings.DEFAULT_LANG

        sample = text[:self.language_sample_size]

        arabic_letters = len(re.findall(
            r'[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]',
            sample
        ))
        latin_letters = len(re.findall(r'[A-Za-z]', sample))

        total_letters = arabic_letters + latin_letters
        if total_letters == 0:
            return self.app_settings.DEFAULT_LANG

        arabic_ratio = arabic_letters / total_letters

        if arabic_ratio >= (1 - self.language_mixed_threshold):
            return "ar"

        if arabic_ratio <= self.language_mixed_threshold:
            return "en"

        return "ar-en"

    async def detect_file_language(self, file_location: str, source_type: str):
        # pdf text needs docling (ocr, images, formulas), which belongs to the
        # processing step, so its language is detected there instead
        try:
            if source_type in self.plain_text_types:
                async with aiofiles.open(file_location, "r", encoding="utf-8", errors="ignore") as f:
                    sample = await f.read(self.language_sample_size)

            elif source_type == ProcessingEnums.DOCX.value:
                # a docx is a zip archive, it has to be unpacked before the text
                # is readable, off the event loop since docx2txt is blocking
                text = await asyncio.to_thread(docx2txt.process, file_location)
                sample = text[:self.language_sample_size] if text else ""

            else:
                return self.app_settings.DEFAULT_LANG

        except Exception:
            return self.app_settings.DEFAULT_LANG

        return self.detect_language(text=sample)

    def _get_clean_file_name(self, orig_file_name: str):

        # remove any special characters, except underscore and .
        cleaned_file_name = re.sub(r'[^\w.]', '', orig_file_name.strip())

        # replace spaces with underscore
        cleaned_file_name = cleaned_file_name.replace(" ", "_")

        return cleaned_file_name