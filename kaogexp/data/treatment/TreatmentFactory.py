from kaogexp.data.treatment.DatasetTreatment import DatasetTreatment
from kaogexp.data.treatment.TreatmentAbstract import TreatmentAbstract


class TreatmentFactory:

    @staticmethod
    def create(type_: str, **kwargs) -> TreatmentAbstract:
        if type_ == "DatasetTreatment":
            return DatasetTreatment(kwargs['dataset'])
