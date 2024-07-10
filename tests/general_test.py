from proby import point


def test_point():
    @point
    def my_point(x):
        return x
