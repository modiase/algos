from solution import main


def test_given():
    """
    given: "dir\\n\\tsubdir1\\n\\t\\tfile1.ext\\n\\t\\tsubsubdir1\\n\\tsubdir2\\n\\t\\tsubsubdir2\\n\\t\\t\\tfile2.ext"
    return ("dir/subdir2/subsubdir2/file2.ext",32)
    """
    res = main(
        "dir\\n\\tsubdir1\\n\\t\\tfile1.ext\\n\\t\\tsubsubdir1\\n\\tsubdir2\\n\\t\\tsubsubdir2\\n\\t\\t\\tfile2.ext")
    assert res[0] == "dir/subdir2/subsubdir2/file2.ext"
    assert res[1] == 32
