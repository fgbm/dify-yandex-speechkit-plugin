import logging
from typing import Any, Generator

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

logger = logging.getLogger(__name__)


class TextToSpeechTool(Tool):
    """
    Yandex SpeechKit Text-to-Speech tool with advanced parameter control
    """

    # Voice descriptions for better UX
    VOICE_DESCRIPTIONS = {
        "marina": "Марина - женский голос (по умолчанию)",
        "alena": "Алёна - женский голос",
        "filipp": "Филипп - мужской голос",
        "jane": "Джейн - женский голос",
        "omazh": "Омаж - мужской голос",
        "ermil": "Ермил - мужской голос",
        "zahar": "Захар - мужской голос",
        "madi_ru": "Мади - мужской голос (рус.)",
    }

    # Allowed emotions per voice (API v1 capabilities)
    VOICE_ALLOWED_EMOTIONS = {
        "marina": {"neutral", "whisper", "friendly"},
        "alena": {"neutral", "good"},
        "filipp": {"neutral"},
        "jane": {"neutral", "good", "evil"},
        "omazh": {"neutral", "evil"},
        "ermil": {"neutral", "good"},
        "zahar": {"neutral", "good"},
        "madi_ru": {"neutral"},
    }

    FORMAT_MIME = {
        "mp3": "audio/mpeg",
        "oggopus": "audio/ogg",
    }

    def _validate_parameters(self, tool_parameters: dict[str, Any]) -> dict[str, str]:
        """
        Validate and process tool parameters
        """
        errors = []

        text = tool_parameters.get("text", "").strip()
        if not text:
            errors.append("Text content is required")
        elif len(text) > 5000:
            errors.append("Text is too long (maximum 5000 characters)")

        voice = tool_parameters.get("voice", "marina")
        if voice not in self.VOICE_DESCRIPTIONS:
            errors.append(f"Invalid voice: {voice}")

        emotion = tool_parameters.get("emotion", "neutral")
        if emotion not in ["neutral", "good", "evil", "friendly", "whisper"]:
            errors.append(f"Invalid emotion: {emotion}")

        try:
            speed = float(tool_parameters.get("speed", 1.0))
            if not (0.1 <= speed <= 3.0):
                errors.append("Speed must be between 0.1 and 3.0")
        except (ValueError, TypeError):
            errors.append("Speed must be a valid number")

        format_type = tool_parameters.get("format", "mp3")
        # API v1 expects 'oggopus' for Opus output
        if format_type == "opus":
            format_type = "oggopus"
        if format_type not in ["mp3", "oggopus"]:
            errors.append(f"Invalid format: {format_type}")

        if errors:
            raise ValueError("; ".join(errors))

        # Validate emotion support for selected voice
        allowed = self.VOICE_ALLOWED_EMOTIONS.get(voice, {"neutral"})
        if emotion not in allowed:
            allowed_list = ", ".join(sorted(allowed))
            errors.append(
                f"Emotion '{emotion}' is not supported by voice '{voice}'. Allowed: {allowed_list}"
            )

        if errors:
            raise ValueError("; ".join(errors))

        return {
            "text": text,
            "voice": voice,
            "emotion": emotion,
            "speed": str(speed),
            "format": format_type,
        }

    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        Invoke the text-to-speech tool
        """
        try:
            # Get API key from credentials
            api_key = self.runtime.credentials.get("api_key")
            if not api_key:
                raise Exception("API key is required")

            # Validate and process parameters
            try:
                params = self._validate_parameters(tool_parameters)
            except ValueError as e:
                raise Exception(f"Parameter error: {str(e)}")

            # Log synthesis parameters
            logger.info(
                f"TTS synthesis - text: '{params['text'][:50]}...', "
                f"voice: {params['voice']}, emotion: {params['emotion']}, "
                f"speed: {params['speed']}, format: {params['format']}"
            )

            # Prepare API request
            url = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
            headers = {"Authorization": f"Api-Key {api_key}"}

            # Prepare request data
            data = {
                # SSML support for v1: if text looks like SSML, use 'ssml' param
                (
                    "ssml" if params["text"].lstrip().startswith("<speak") else "text"
                ): params["text"],
                "voice": params["voice"],
                "speed": params["speed"],
                "format": params["format"],
                "lang": "ru-RU",
            }
            # Add emotion only if not neutral or explicitly supported
            if params["emotion"] != "neutral":
                data["emotion"] = params["emotion"]

            # Make API request
            response = requests.post(url, headers=headers, data=data, timeout=30)

            logger.info(f"API response status: {response.status_code}")

            if response.status_code != 200:
                error_msg = f"API error: {response.status_code}"
                try:
                    error_data = response.json()
                    # Normalize different shapes
                    nested = (
                        error_data.get("error")
                        if isinstance(error_data, dict)
                        else None
                    )
                    if isinstance(nested, dict):
                        code = nested.get("code") or nested.get("error_code")
                        msg = nested.get("message") or nested.get("error_message")
                    else:
                        code = error_data.get("error_code") or error_data.get("code")
                        msg = error_data.get("message") or error_data.get(
                            "error_message"
                        )
                    # Fallback to details if present
                    if not msg:
                        details = error_data.get("details")
                        if isinstance(details, (list, dict)):
                            msg = str(details)[:200]
                    if code:
                        error_msg += f" [{code}]"
                    if msg:
                        error_msg += f" - {msg}"
                except Exception:
                    body = (response.text or "")[:200]
                    error_msg += f" - {body}"

                # Append params context to help debugging
                error_msg += (
                    f" | voice={params['voice']}, emotion={params['emotion']}, "
                    f"speed={params['speed']}, format={params['format']}"
                )

                logger.error(error_msg)
                raise Exception(error_msg)

            # Get audio content
            audio_content = response.content
            if not audio_content:
                raise Exception("Empty response from TTS service")

            logger.info(
                f"TTS synthesis successful, audio size: {len(audio_content)} bytes"
            )

            # Create result message
            result_text = (
                f"Speech synthesis completed successfully!\n\n"
                f"**Parameters used:**\n"
                f"• Voice: {params['voice']} ({self.VOICE_DESCRIPTIONS.get(params['voice'], 'Unknown')})\n"
                f"• Emotion: {params['emotion']}\n"
                f"• Speed: {params['speed']}x\n"
                f"• Format: {params['format'].upper()}\n"
                f"• Text length: {len(params['text'])} characters\n"
                f"• Audio size: {len(audio_content)} bytes"
            )

            # Return structured variables: first named variable 'params', then file with 'file' name
            yield self.create_json_message(
                {
                    "voice": params["voice"],
                    "emotion": params["emotion"],
                    "speed": params["speed"],
                    "format": (
                        "opus" if params["format"] == "oggopus" else params["format"]
                    ),
                    "text_length": len(params["text"]),
                },
            )

            # Return file as blob (goes to default 'files' array)
            ext = "ogg" if params["format"] == "oggopus" else params["format"]
            yield self.create_blob_message(
                blob=audio_content,
                meta={
                    "mime_type": self.FORMAT_MIME.get(
                        params["format"], "application/octet-stream"
                    ),
                    "filename": f"speech.{ext}",
                },
            )

        except requests.exceptions.Timeout:
            logger.error("TTS request timeout")
            raise Exception("Request timeout. Please try again with shorter text.")
        except requests.exceptions.RequestException as e:
            logger.error(f"TTS request error: {str(e)}")
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            raise
