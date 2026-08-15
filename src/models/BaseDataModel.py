from helpers import get_settings, Settings

class BaseDataModel:

    def __init__(self, db_client: object):
        self.db_client = db_client
        self.settings: Settings = get_settings()

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        return instance

