import pytest
import shutil

from src.configuration import *


@pytest.fixture(scope="class")
def setup_test_output():
    os.mkdir("src/unittests/test_output_dump")


class TestLoadSettings:
    def test_file_found(self):
        settings = load_settings("src/appsettings.json")
        assert len(settings) != 0
    
    def test_file_not_found(self):
        with pytest.raises(UserWarning):
            load_settings("bad-filename")


class TestExtractFile:
    def teardown_class(self):
        if os.path.exists("src/unittests/test_output_dump"):
            shutil.rmtree('src/unittests/test_output_dump', ignore_errors=True)

    def test_extract_filenotfound(self):
        with pytest.raises(UserWarning):
            extract_file("some.zip", "src/unittests/test_output_dump")

    def test_extract_badfile(self):
        with pytest.raises(UserWarning):
            extract_file("src/unittests/test_input_dir/bad_input_zipped.zip", "src/unittests/test_output_dump")

    def test_extract_filefound(self):
        try:
            extract_file("src/unittests/test_input_dir/correct_input_to_unzip.zip", "src/unittests/test_output_dump")
        except UserWarning as exc:
            assert False, exc


