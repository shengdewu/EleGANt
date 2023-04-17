from . import transformer_type
from ..post import config


class Compose(object):
    """Compose multiple transforms sequentially.
    """

    def __init__(self):
        transforms = config.transforms
        assert isinstance(transforms, list)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                kwargs = config.get_kwargs(transform, typename=transform['typename'])
                transform = transformer_type[transform['typename']](**kwargs)
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
