from kaogexp.data.normalizer.MinMaxNormalizer import MinMaxNormalizer


class NormalizerFactory:

    @staticmethod
    def create(type: str, **kwargs):
        if type == "MinMaxNormalizer":
            return MinMaxNormalizer(**kwargs)
