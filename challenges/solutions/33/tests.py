import pytest

from .solution import main
from .utils import Arguments


def test_inputting_race_gives_ecarace():

    args = Arguments(word='race')

    result = main(args)

    assert result == 'ecarace'


if __name__ == '__main__':
    pytest.main()