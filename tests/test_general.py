from unittest import TestCase


class Test(TestCase):

    def test11(self):
        from importlib.resources import files
        # Reads contents with UTF-8 encoding and returns str.
        print(files("synthwave.data"))


    def test_get_formatted_attributes(self):

        self.fail()
