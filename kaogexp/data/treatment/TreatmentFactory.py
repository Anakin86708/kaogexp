from kaogexp.data.treatment.DatasetTreatment import DatasetTreatment
from kaogexp.data.treatment.TreatmentAbstract import TreatmentAbstract


class TreatmentFactory:

    @staticmethod
    def create(type: str, **kwargs) -> TreatmentAbstract:
        if type == "DatasetTreatment":
            return DatasetTreatment(kwargs)
