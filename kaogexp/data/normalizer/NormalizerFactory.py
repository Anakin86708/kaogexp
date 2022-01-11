from kaogexp.data.normalizer.MinMaxNormalizer import MinMaxNormalizer


class NormalizerFactory:

    @staticmethod
    def create(type_: str, **kwargs):
        if type_ == "MinMaxNormalizer":
            return MinMaxNormalizer(**kwargs)
