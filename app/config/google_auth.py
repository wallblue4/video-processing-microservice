# app/config/google_auth.py - 🆕 NUEVO (tomado del microservicio clasificación)
import os
import json
import tempfile
import logging

logger = logging.getLogger(__name__)

def setup_google_credentials():
    """
    Configurar credenciales de Google Cloud en Render usando JSON en variable de entorno
    """
    
    # Obtener JSON de credenciales desde variable de entorno
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    
    if credentials_json:
        try:
            # Crear archivo temporal con las credenciales
            credentials_dict = json.loads(credentials_json)
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                json.dump(credentials_dict, f)
                credentials_file = f.name
            
            # Configurar variable de entorno para Google Cloud
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_file
            
            logger.info("✅ Credenciales Google Cloud configuradas desde JSON")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error configurando credenciales desde JSON: {e}")
            return False
    else:
        # Fallback: verificar si ya existe GOOGLE_APPLICATION_CREDENTIALS (desarrollo local)
        existing_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if existing_creds and os.path.exists(existing_creds):
            logger.info("✅ Usando credenciales Google Cloud existentes (desarrollo local)")
            return True
        
        logger.warning("⚠️ GOOGLE_APPLICATION_CREDENTIALS_JSON no configurado")
        return False