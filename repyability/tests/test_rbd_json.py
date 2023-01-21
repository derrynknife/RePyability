import surpyval as surv
from repyability.rbd.non_repairable_rbd import NonRepairableRBD
import json
from repyability import __version__ as repyability_version

ACTUAL_OUTPUT_DIR = "./rbd_json/actual_output"
EXP_OUTPUT_DIR = "./rbd_json/exp_output"


def assert_nonrepairable_rbd_json(exp_filename: str, actual_filename: str):
    """Asserts exp_filename json and actual_filename json represent the same
    RBD.

    This is with the exception that it doesn't check exp_filename's
    `repyability_version` key/value pair, but that actual_filename's
    `repyability_version` is correct to the repyability version being run.

    Parameters
    ----------
    exp_filename : str
        The filename of the expected output JSON file
    actual_filename : str
        The filename of the actual output JSON file
    """
    # Open the expected and actual JSON files
    with open(exp_filename) as exp_f:
        exp_dict = json.load(exp_f)

    with open(actual_filename) as actual_f:
        actual_dict = json.load(actual_f)

    # Assert repyability version
    assert actual_dict["repyability_version"] == repyability_version

    # Assert edges array
    for i in range(len(exp_dict["edges"])):
        assert exp_dict["edges"][i] == actual_dict["edges"][i]

    # Assert reliabilities array
    for i in range(len(exp_dict["reliabilities"])):
        assert exp_dict["reliabilities"][i] == actual_dict["reliabilities"][i]


def test_nonrepairable_rbd_json_save():
    """
    Creates and saves a NonRepairableRBD to
    ./rbd_json/actual_output/test_nonrepairable_rbd_json_save.json and
    asserts this is the same as
    ./rbd_json/exp_output/test_nonrepairable_rbd_json_save.json,
    taking into consideration repyability's version name/value pair.
    """
    exp_filename = f"{EXP_OUTPUT_DIR}/test_nonrepairable_rbd_json_save.json"
    actual_filename = (
        f"{ACTUAL_OUTPUT_DIR}/test_nonrepairable_rbd_json_save.json"
    )

    edges = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
    ]
    reliabilities = {
        2: surv.Weibull.from_params([10, 2]),
        3: surv.Weibull.from_params([10, 2]),
        4: surv.Weibull.from_params([10, 2]),
        5: surv.Weibull.from_params([10, 2]),
    }

    rbd = NonRepairableRBD(edges, reliabilities)

    rbd.save_to_json(actual_filename)

    assert_nonrepairable_rbd_json(exp_filename, actual_filename)


def test_nonrepairable_rbd_json_save_rbd_parallel(
    rbd_parallel: NonRepairableRBD,
):
    exp_filename = (
        f"{EXP_OUTPUT_DIR}/test_nonrepairable_rbd_json_save_rbd_parallel.json"
    )
    actual_filename = f"{ACTUAL_OUTPUT_DIR}/\
        test_nonrepairable_rbd_json_save_rbd_parallel.json"

    rbd_parallel.save_to_json(actual_filename)

    assert_nonrepairable_rbd_json(exp_filename, actual_filename)
