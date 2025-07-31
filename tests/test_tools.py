import pytest
from unittest.mock import Mock
from agentic import tools

def test_save_memory(mocker):
    # Mock os.getcwd to have a predictable project name
    mocker.patch("os.getcwd", return_value="/path/to/my_project")
    # Mock the config helpers
    mocker.patch("agentic.config._ensure_data_dir")
    mocker.patch("pathlib.Path.open", mocker.mock_open())
    mocker.patch.object(tools.config, 'DATA_DIR', tools.Path('/fake/data/dir'))
    
    # Test project scope (default)
    result_project = tools.save_memory("info")
    assert result_project == "OK, I will remember this for future 'project' sessions."
    tools.config.DATA_DIR.joinpath("my_project.md").open.assert_called_once_with("a", encoding="utf-8")
    
    tools.config.DATA_DIR.joinpath("my_project.md").open.reset_mock()
    
    # Test global scope
    result_global = tools.save_memory("global info", scope="global")
    assert result_global == "OK, I will remember this for future 'global' sessions."
    tools.config.DATA_DIR.joinpath("memorys.global.md").open.assert_called_once_with("a", encoding="utf-8")

def test_think():
    assert tools.think("a thought") == "Thought successfully processed!"

def test_file_operations(tmp_path):
    """Tests WriteFile, ReadFile, Edit, ReadFolder, FindFiles, and SearchText."""
    # Setup
    test_dir = tmp_path / "test_project"
    test_dir.mkdir()
    file1_path = test_dir / "file1.txt"
    subdir = test_dir / "subdir"
    subdir.mkdir()
    file2_path = subdir / "file2.py"
    read_files_in_session = set()

    # 1. WriteFile
    write_result = tools.write_file(str(file1_path), "Hello\nWorld")
    assert "Successfully wrote" in write_result
    assert file1_path.read_text() == "Hello\nWorld"

    tools.write_file(str(file2_path), "import os")

    # 2. ReadFile
    content = tools.read_file(str(file1_path), read_files_in_session)
    assert content == "Hello\nWorld"
    assert str(file1_path) in read_files_in_session

    # Test reading a non-existent file
    read_error = tools.read_file(str(test_dir / "no.txt"), read_files_in_session)
    assert "Error: File not found" in read_error

    # 3. Edit
    edit_result = tools.edit(str(file1_path), "World", "Universe")
    assert "Successfully edited" in edit_result
    assert file1_path.read_text() == "Hello\nUniverse"
    
    # Test editing with a non-existent search string
    edit_error = tools.edit(str(file1_path), "Foo", "Bar")
    assert "ERROR: Search string not found" in edit_error

    # 4. ReadFolder
    folder_content = tools.read_folder(str(test_dir))
    assert "file1.txt" in folder_content
    assert "subdir/" in folder_content

    # 5. FindFiles
    found_files = tools.find_files(f"{test_dir}/**/*.py")
    assert str(file2_path) in found_files

    # 6. SearchText
    search_result = tools.search_text("Hello", str(file1_path))
    assert "1:Hello" in search_result

def test_shell_command():
    """Tests the shell tool with a safe command."""
    result = tools.shell('echo "test"')
    assert "test" in result
    assert "STDOUT" in result

def test_web_fetch(mocker):
    """Tests the web_fetch tool by mocking requests."""
    # Mock successful request
    mock_response_ok = Mock()
    mock_response_ok.text = "<html>Success</html>"
    mock_response_ok.raise_for_status.return_value = None
    mocker.patch("requests.get", return_value=mock_response_ok)
    
    result = tools.web_fetch("http://example.com")
    assert result == "<html>Success</html>"

    # Mock failed request
    mock_response_fail = Mock()
    mock_response_fail.raise_for_status.side_effect = Exception("HTTP Error")
    mocker.patch("requests.get", return_value=mock_response_fail)
    
    error_result = tools.web_fetch("http://example.com/fail")
    assert "Error fetching URL" in error_result
