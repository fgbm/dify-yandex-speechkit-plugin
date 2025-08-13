import logging
from io import BytesIO
from typing import Any, Generator

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

logger = logging.getLogger(__name__)


class SpeechToTextTool(Tool):
    """
    Yandex SpeechKit Speech-to-Text tool with support for multiple audio formats including WEBM
    """

    SUPPORTED_EXTENSIONS = [
        ".mp3",
        ".wav",
        ".flac",
        ".ogg",
        ".opus",
        ".m4a",
        ".webm",
        ".aac",
        ".wma",
    ]

    def _get_audio_data_from_file(self, audio_file: Any) -> bytes:
        """
        Extract audio data from various file formats (Dify file object, file-like object, or bytes)
        """
        try:
            logger.info(f"Processing audio file of type: {type(audio_file)}")

            # Case 1: Dify File object (dify_plugin.file.file.File)
            # Prefer explicit download() to avoid lazy URL fetch inside .blob
            if hasattr(audio_file, "download") or hasattr(audio_file, "blob"):
                logger.info("Processing Dify File object via download/blob")
                # Prefer download()
                if hasattr(audio_file, "download"):
                    content = audio_file.download()
                    if isinstance(content, (bytes, bytearray, memoryview)):
                        return bytes(content)
                    # Some SDKs may return an object with 'content' attribute
                    if hasattr(content, "content"):
                        return content.content
                # Fallback: blob/read/content/data
                if hasattr(audio_file, "blob"):
                    content = audio_file.blob
                    if isinstance(content, (bytes, bytearray, memoryview)):
                        return bytes(content)
                if hasattr(audio_file, "read"):
                    return audio_file.read()
                if hasattr(audio_file, "content"):
                    return audio_file.content
                if hasattr(audio_file, "data"):
                    return audio_file.data

                # Fallback 2: try SDK attributes or absolute URL without requiring env vars
                # 2a) Try absolute remote_url if provided
                remote_url = None
                for attr_name in ("remote_url", "url"):
                    if hasattr(audio_file, attr_name):
                        remote_url = getattr(audio_file, attr_name)
                        break
                    if isinstance(getattr(audio_file, "__dict__", {}), dict):
                        remote_url = audio_file.__dict__.get(attr_name) or remote_url

                if isinstance(remote_url, str) and remote_url.lower().startswith(
                    ("http://", "https://")
                ):
                    try:
                        logger.info(f"Fetching file via absolute URL: {remote_url}")
                        resp = requests.get(remote_url, timeout=30)
                        resp.raise_for_status()
                        return resp.content
                    except Exception as url_err:
                        logger.warning(f"Failed to fetch via absolute URL: {url_err}")

                # 2b) Try ID-based retrieval if available on runtime/session (SDK-dependent)
                try:
                    file_id = None
                    for attr_name in ("id", "related_id"):
                        if hasattr(audio_file, attr_name):
                            file_id = getattr(audio_file, attr_name)
                            if file_id:
                                break
                        if (
                            isinstance(getattr(audio_file, "__dict__", {}), dict)
                            and not file_id
                        ):
                            file_id = audio_file.__dict__.get(attr_name) or file_id

                    if file_id:
                        if hasattr(self, "get_file_content"):
                            return self.get_file_content(file_id)
                        if hasattr(self.runtime, "get_file_content"):
                            return self.runtime.get_file_content(file_id)
                        if hasattr(self.runtime, "get_file"):
                            return self.runtime.get_file(file_id)
                except Exception as id_err:
                    logger.warning(f"Failed to fetch via SDK by id: {id_err}")

                # Nothing else worked
                raise ValueError(
                    "Could not read Dify File object: no blob/download/read/data/content available, "
                    "no absolute URL, and SDK file fetch by id not supported in this runtime."
                )

            # Case 2: File-like object (has read method)
            elif hasattr(audio_file, "read"):
                logger.info("Processing file-like object with read() method")
                content = audio_file.read()
                logger.info(
                    f"Successfully read file content, size: {len(content)} bytes"
                )
                return content

            # Case 3: Already bytes
            elif isinstance(audio_file, bytes):
                logger.info("Processing raw bytes data")
                return audio_file

            # Case 4: String path to file
            elif isinstance(audio_file, str):
                logger.info(f"Processing file path: {audio_file}")
                with open(audio_file, "rb") as f:
                    return f.read()

            # Case 5: Dictionary with file metadata (Dify passes this in some contexts)
            elif isinstance(audio_file, dict):
                logger.info(f"Processing file metadata dict: {audio_file}")
                # Prefer fetching by file id via SDK/session if available
                related_id = audio_file.get("related_id") or audio_file.get("id")
                if related_id:
                    # Try a series of SDK methods that may exist depending on runtime
                    for accessor in (
                        getattr(self, "get_file_content", None),
                        getattr(
                            getattr(self, "runtime", None), "get_file_content", None
                        ),
                        getattr(getattr(self, "runtime", None), "get_file", None),
                        getattr(
                            getattr(self, "session", None), "get_file_content", None
                        ),
                        getattr(getattr(self, "session", None), "get_file", None),
                        getattr(
                            getattr(getattr(self, "session", None), "files", None),
                            "get",
                            None,
                        ),
                        getattr(
                            getattr(getattr(self, "session", None), "file", None),
                            "get",
                            None,
                        ),
                    ):
                        try:
                            if callable(accessor):
                                res = accessor(related_id)
                                # Some accessors may return an object; try common attributes
                                if isinstance(res, (bytes, bytearray)):
                                    return bytes(res)
                                if hasattr(res, "blob"):
                                    return res.blob
                                if hasattr(res, "content"):
                                    return res.content
                                if hasattr(res, "download"):
                                    return res.download()
                        except Exception as sdk_err:
                            logger.debug(f"SDK fetch attempt failed: {sdk_err}")

                # As a last resort, only accept absolute URLs
                remote_url = audio_file.get("remote_url") or audio_file.get("url")
                if isinstance(remote_url, str) and remote_url.lower().startswith(
                    ("http://", "https://")
                ):
                    try:
                        logger.info(f"Fetching file via absolute URL: {remote_url}")
                        resp = requests.get(remote_url, timeout=30)
                        resp.raise_for_status()
                        return resp.content
                    except Exception as url_err:
                        logger.debug(f"Absolute URL fetch failed: {url_err}")

                # If we reach here, we could not retrieve bytes safely
                raise ValueError(
                    "Could not retrieve file bytes from provided metadata (no SDK method worked, "
                    "and no absolute URL available)."
                )

            else:
                # List available attributes for debugging
                if hasattr(audio_file, "__dict__") or hasattr(audio_file, "__class__"):
                    attrs = [
                        attr for attr in dir(audio_file) if not attr.startswith("_")
                    ]
                    logger.info(f"Available attributes: {attrs}")

                raise ValueError(
                    f"Unsupported audio file type: {type(audio_file)}. "
                    f"Expected Dify File object with .blob property."
                )

        except Exception as e:
            logger.error(f"Error extracting audio data: {str(e)}")
            raise

    def _convert_audio_to_wav(
        self, audio_data: bytes, source_format: str = None
    ) -> bytes:
        """
        Convert audio data to WAV format using pydub
        """
        try:
            from pydub import AudioSegment
            from pydub.utils import which

            # Check if ffmpeg is available
            if not which("ffmpeg"):
                logger.warning("FFmpeg not found, attempting conversion without it")

            # Load audio from bytes
            audio = AudioSegment.from_file(BytesIO(audio_data))

            # Convert to mono 16kHz WAV (required by Yandex)
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_frame_rate(16000)  # 16kHz
            audio = audio.set_sample_width(2)  # 16-bit

            # Export to WAV bytes
            wav_buffer = BytesIO()
            audio.export(wav_buffer, format="wav")
            return wav_buffer.getvalue()

        except ImportError:
            logger.error(
                "pydub library is not installed. Please install it: pip install pydub"
            )
            raise ValueError("Audio conversion library not available")
        except Exception as e:
            logger.error(f"Audio conversion failed: {str(e)}")
            raise ValueError(f"Failed to convert audio: {str(e)}")

    def _convert_wav_to_pcm(self, wav_data: bytes) -> tuple[bytes, int]:
        """
        Convert WAV data to raw PCM and extract sample rate
        """
        try:
            import audioop
            import wave

            with wave.open(BytesIO(wav_data), "rb") as wav_reader:
                src_rate = wav_reader.getframerate()
                sample_width = wav_reader.getsampwidth()
                channels = wav_reader.getnchannels()
                frames = wav_reader.getnframes()

                if sample_width != 2:
                    raise ValueError(
                        f"Unsupported WAV sample width: {sample_width * 8} bits. Only 16-bit PCM supported"
                    )

                raw = wav_reader.readframes(frames)

                # Convert to mono if needed
                if channels > 1:
                    raw = audioop.tomono(raw, sample_width, 0.5, 0.5)

                # Apply gain normalization for quiet recordings
                try:
                    rms = audioop.rms(raw, sample_width)
                    gain = 1.0
                    if rms and rms < 300:
                        gain = 4.0
                    elif rms and rms < 1000:
                        gain = 2.0
                    if gain != 1.0:
                        raw = audioop.mul(raw, sample_width, gain)
                        logger.info(f"Applied gain x{gain}, previous RMS={rms}")
                except Exception:
                    pass

                return raw, src_rate

        except Exception as e:
            logger.error(f"WAV to PCM conversion failed: {str(e)}")
            raise ValueError(f"Failed to process WAV file: {str(e)}")

    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        Invoke the speech-to-text tool
        """
        try:
            # Get parameters
            audio_file = tool_parameters.get("audio_file")
            language = tool_parameters.get("language", "ru-RU")
            topic = tool_parameters.get("topic", "general")

            if not audio_file:
                raise Exception("No audio file provided")

            # Get API key from credentials
            api_key = self.runtime.credentials.get("api_key")
            if not api_key:
                raise Exception("API key is required")

            # Read audio data
            try:
                audio_data = self._get_audio_data_from_file(audio_file)
                if not audio_data:
                    raise Exception("Could not read audio file")
            except Exception as e:
                logger.error(f"Failed to read audio file: {str(e)}")
                raise Exception(f"Failed to read audio file - {str(e)}")

            logger.info(f"Processing audio file, size: {len(audio_data)} bytes")

            # Check if conversion is needed
            # For efficiency, we'll try to detect if it's already a WAV file
            try:
                # Try to process as WAV directly
                pcm_data, sample_rate = self._convert_wav_to_pcm(audio_data)
                logger.info("Audio processed as WAV directly")
            except:
                # If that fails, convert from other format first
                logger.info("Converting audio to WAV format")
                wav_data = self._convert_audio_to_wav(audio_data)
                pcm_data, sample_rate = self._convert_wav_to_pcm(wav_data)
                logger.info("Audio converted and processed successfully")

            # Prepare API request
            url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
            headers = {"Authorization": f"Api-Key {api_key}"}
            params = {
                "lang": language,
                "format": "lpcm",
                "topic": topic,
                "sampleRateHertz": str(sample_rate),
            }

            logger.info(f"Making API request with params: {params}")

            # Send request to Yandex SpeechKit
            response = requests.post(
                url, headers=headers, params=params, data=pcm_data, timeout=30
            )

            logger.info(f"API response status: {response.status_code}")

            if response.status_code != 200:
                error_msg = f"API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('error', {}).get('message', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text}"

                logger.error(error_msg)
                raise Exception(error_msg)

            # Parse response
            result = response.json()
            recognized_text = result.get("result", "")

            if not recognized_text:
                raise Exception("No speech detected in the audio file")

            logger.info(f"Recognition successful: {recognized_text[:100]}...")
            yield self.create_text_message(recognized_text)

        except Exception as e:
            # Ensure tool errors are surfaced as failures, not normal text output
            logger.error(f"Speech recognition error: {str(e)}")
            raise
