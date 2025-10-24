"""Security Manager for data protection and privacy features."""

import os
import json
import keyring
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
# FIXED: Import PBKDF2HMAC instead of PBKDF2
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
from loguru import logger
# Added for keyring error handling
from keyring.errors import PasswordDeleteError


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    encrypt_storage: bool = True
    encryption_algorithm: str = "AES-128-CBC" # Fernet uses AES-128-CBC with HMAC-SHA256
    data_retention_days: int = 7
    auto_cleanup_enabled: bool = True
    store_api_keys: bool = False
    api_key_encryption: bool = True
    enable_data_minimization: bool = True
    anonymize_sensitive_data: bool = False
    audit_logging: bool = True
    audit_log_retention_days: int = 90
    enable_gdpr_features: bool = True


class SecurityManager:
    """
    Manages security and privacy features like encryption, key storage, and logging.
    """
    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        storage_dir: Optional[Path] = None,
        master_key: Optional[str] = None,
    ):
        """Initialize SecurityManager."""
        self.config = config or SecurityConfig()
        self.storage_dir = storage_dir or Path("temp/secure")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path = self.storage_dir / "audit.log"
        self.cipher_suite = self._initialize_encryption(master_key)
        logger.info("SecurityManager initialized")
        self._audit_log("security_manager_initialized", {"config": self.config.__dict__})

    def _initialize_encryption(self, master_key: Optional[str] = None) -> Optional[Fernet]:
        """Initialize encryption cipher suite (Fernet)."""
        if not self.config.encrypt_storage:
            logger.info("Encryption disabled by configuration.")
            return None
        try:
            key_path = self.storage_dir / ".encryption_key"
            key: bytes
            if master_key is None:
                # Use or generate a key file if no master key provided
                if key_path.exists():
                    key = key_path.read_bytes()
                    logger.info("Loaded existing encryption key from file.")
                else:
                    key = Fernet.generate_key()
                    key_path.write_bytes(key)
                    key_path.chmod(0o600) # Restrict permissions
                    logger.info("Generated and saved new encryption key file.")
            else:
                # Derive a key from the provided master key string using PBKDF2HMAC
                salt = b'\x00'*16 # Use a securely stored random salt in production!
                kdf = PBKDF2HMAC( # FIXED: Use PBKDF2HMAC here
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=600000, # Recommended iterations
                    backend=default_backend(),
                )
                key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
                logger.info("Derived encryption key from master key using PBKDF2HMAC.")

            cipher_suite = Fernet(key)
            logger.info("Encryption cipher suite initialized successfully.")
            return cipher_suite
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}", exc_info=True)
            # Depending on policy, either raise e or disable encryption
            return None # Fail safe: disable encryption if init fails

    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data using the initialized cipher suite."""
        if not self.cipher_suite or not self.config.encrypt_storage:
            if isinstance(data, str):
                return data.encode('utf-8')
            return data # Return original bytes if encryption is off

        if isinstance(data, str):
            data = data.encode("utf-8")

        try:
            encrypted = self.cipher_suite.encrypt(data)
            self._audit_log("data_encrypted", {"size": len(data)})
            return encrypted
        except Exception as e:
            logger.error(f"Encryption failed: {e}", exc_info=True)
            raise RuntimeError("Data encryption failed.") from e

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using the initialized cipher suite."""
        if not self.cipher_suite or not self.config.encrypt_storage:
            return encrypted_data # Return original bytes if encryption is off

        try:
            decrypted = self.cipher_suite.decrypt(encrypted_data)
            self._audit_log("data_decrypted", {"size": len(decrypted)})
            return decrypted
        except Exception as e: # Catch specific crypto errors like InvalidToken if needed
            logger.error(f"Decryption failed: {e}", exc_info=True)
            raise ValueError("Decryption failed. Key incorrect or data corrupted?") from e

    def store_api_key(self, service_name: str, api_key: str, username: str = "default") -> None:
        """Store API key securely using keyring, potentially encrypting first."""
        if not self.config.store_api_keys:
            logger.warning("API key storage is disabled in configuration.")
            return
        try:
            key_to_store = api_key
            if self.config.api_key_encryption and self.cipher_suite:
                encrypted_key = self.encrypt_data(api_key)
                key_to_store = base64.urlsafe_b64encode(encrypted_key).decode('ascii')
                logger.debug(f"Encrypting API key for {service_name} before storing.")
            elif self.config.api_key_encryption and not self.cipher_suite:
                 logger.error(f"Cannot store encrypted API key for {service_name}: Encryption failed to initialize.")
                 return # Or raise error

            keyring.set_password(f"termextractor_{service_name}", username, key_to_store)
            logger.info(f"API key stored securely via keyring for {service_name}")
            self._audit_log("api_key_stored", {"service": service_name, "username": username})
        except Exception as e: # Catch specific keyring errors if possible
            logger.error(f"Failed to store API key using keyring for {service_name}: {e}", exc_info=True)
            # Consider alternative storage or raise error based on requirements

    def retrieve_api_key(self, service_name: str, username: str = "default") -> Optional[str]:
        """Retrieve stored API key using keyring, decrypting if necessary."""
        try:
            stored_value = keyring.get_password(f"termextractor_{service_name}", username)
            if not stored_value:
                logger.info(f"No API key found in keyring for {service_name}/{username}")
                return None

            api_key: Optional[str] = None
            if self.config.api_key_encryption and self.cipher_suite:
                try:
                    encrypted_bytes = base64.urlsafe_b64decode(stored_value.encode('ascii'))
                    decrypted_bytes = self.decrypt_data(encrypted_bytes)
                    api_key = decrypted_bytes.decode("utf-8")
                    logger.debug(f"Decrypted API key retrieved for {service_name}")
                except (ValueError, TypeError, base64.binascii.Error) as dec_e: # Catch specific decode/decrypt errors
                    logger.error(f"Failed to decode/decrypt stored API key for {service_name}: {dec_e}. Keyring value might not be encrypted or is corrupted.")
                    # Fallback decision: Return None? Return raw? Log error and return None?
                    return None # Safer option
                except Exception as e:
                     logger.error(f"Unexpected error during API key decryption for {service_name}: {e}", exc_info=True)
                     return None
            else:
                # Assumes stored value is the raw API key if encryption is off or failed
                api_key = stored_value
                logger.debug(f"Retrieved raw API key for {service_name} (encryption off or failed init)")

            if api_key:
                 self._audit_log("api_key_retrieved", {"service": service_name, "username": username})
            return api_key
        except Exception as e: # Catch specific keyring errors if possible
            logger.error(f"Failed to retrieve API key using keyring for {service_name}: {e}", exc_info=True)
            return None

    def delete_api_key(self, service_name: str, username: str = "default") -> None:
        """Delete stored API key using keyring."""
        try:
            keyring.delete_password(f"termextractor_{service_name}", username)
            logger.info(f"API key deleted from keyring for {service_name}/{username}")
            self._audit_log("api_key_deleted", {"service": service_name, "username": username})
        except PasswordDeleteError:
             logger.info(f"API key for {service_name}/{username} not found in keyring, nothing to delete.")
        except Exception as e: # Catch specific keyring errors
            logger.warning(f"Failed to delete API key using keyring for {service_name}: {e}", exc_info=True)

    def _audit_log(self, event: str, details: Dict[str, Any]) -> None:
        """Log audit event if enabled."""
        if not self.config.audit_logging:
            return
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "details": details,
        }
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                # Use default=str to handle non-serializable objects like Path
                f.write(json.dumps(log_entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log entry: {e}")

    # --- Other methods (placeholders need implementation based on requirements) ---

    def minimize_data(self, text: str, max_context_chars: int = 1000) -> str:
        """Placeholder for data minimization logic."""
        if self.config.enable_data_minimization and len(text) > max_context_chars:
             # Simple truncation example
             half = max_context_chars // 2
             minimized = text[:half] + "\n[...TRUNCATED...]\n" + text[-half:]
             logger.debug(f"Minimized data from {len(text)} to {len(minimized)} chars.")
             self._audit_log("data_minimized", {"original_size": len(text), "minimized_size": len(minimized)})
             return minimized
        return text

    def anonymize_text(self, text: str) -> str:
        """Placeholder for sensitive data anonymization logic."""
        if not self.config.anonymize_sensitive_data:
            return text
        # Add actual anonymization logic using regex or NLP tools here
        logger.warning("Anonymization enabled but not implemented yet.")
        self._audit_log("data_anonymization_skipped", {"reason": "Not implemented"})
        return text # Return original for now

    def cleanup_old_data(self) -> int:
        """Placeholder for data cleanup logic based on retention."""
        if not self.config.auto_cleanup_enabled:
            return 0
        # Add logic to find and delete files older than config.data_retention_days
        logger.warning("Auto cleanup enabled but not implemented yet.")
        self._audit_log("data_cleanup_skipped", {"reason": "Not implemented"})
        return 0

    def get_audit_log(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, event_type: Optional[str] = None) -> list:
        """Placeholder for retrieving audit log entries."""
        if not self.config.audit_logging or not self.audit_log_path.exists():
            return []
        # Add logic to read and filter log file
        logger.warning("Audit log retrieval not fully implemented yet.")
        return []

    def export_data(self, user_id: str, output_path: Path) -> None:
        """Placeholder for GDPR data export."""
        if not self.config.enable_gdpr_features:
            logger.warning("GDPR features disabled, cannot export data.")
            return
        # Add logic to collect and write user data
        logger.warning(f"Data export for user {user_id} not implemented yet.")
        self._audit_log("data_export_skipped", {"user_id": user_id, "reason": "Not implemented"})

    def delete_user_data(self, user_id: str) -> None:
        """Placeholder for GDPR data deletion."""
        if not self.config.enable_gdpr_features:
            logger.warning("GDPR features disabled, cannot delete data.")
            return
        # Add logic to find and delete user data
        logger.warning(f"Data deletion for user {user_id} not implemented yet.")
        self._audit_log("data_deletion_skipped", {"user_id": user_id, "reason": "Not implemented"})
