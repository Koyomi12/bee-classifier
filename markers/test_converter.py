from converter import Converter


def test_convert_from_a_to_b():
    converter = Converter((0, 0), (1000, 0), (0, 0), (500, 0), 39)
    assert converter.convert_from_a_to_b((250, 250)) == (125, 125), (
        "Should be (125, 125)"
    )


def test_convert_from_a_to_b2():
    converter = Converter((0, 0), (-1000, 0), (0, 0), (-500, 0), 39)
    assert converter.convert_from_a_to_b((250, 250)) == (125, 125), (
        "Should be (125, 125)"
    )


def test_convert_from_a_to_b3():
    converter = Converter((0, 0), (-1000, 0), (0, 0), (500, 0), 39)
    assert converter.convert_from_a_to_b((250, 250)) == (125, 125), (
        "Should be (125, 125)"
    )


def test_convert_from_a_to_b4():
    converter = Converter((0, 0), (-1000, 0), (0, 0), (500, 0), 39)
    assert converter.convert_from_a_to_b((250, -250)) == (125, -125), (
        "Should be (125, 125)"
    )


if __name__ == "__main__":
    test_convert_from_a_to_b()
    test_convert_from_a_to_b2()
    test_convert_from_a_to_b3()
    test_convert_from_a_to_b4()
    print("Everything passed")
