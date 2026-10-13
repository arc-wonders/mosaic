"""Skeleton geometry: rule evaluation, unknowns, signatures, and the session tools built on them."""

import numpy as np
import pytest

from mosaic.geometry import DEFAULT_RULES, RuleBook, Skeleton, fit_signature, owner, part_features, signature_loglik
from mosaic.pose import KEYPOINTS
from test_capability import session  # noqa: F401  (fixture)

# A person facing the camera (their left is the image right), 300 px tall. Left arm in an "L":
# upper arm straight down, forearm turned out to the image right. Right arm hanging.
JOINTS = {
    "nose": (150, 40), "left_eye": (158, 32), "right_eye": (142, 32), "left_ear": (168, 38), "right_ear": (132, 38),
    "left_shoulder": (180, 80), "right_shoulder": (120, 80), "left_elbow": (182, 140), "right_elbow": (118, 140),
    "left_wrist": (240, 142), "right_wrist": (116, 200), "left_hip": (170, 180), "right_hip": (130, 180),
    "left_knee": (172, 240), "right_knee": (128, 240), "left_ankle": (173, 300), "right_ankle": (127, 300),
}
BOX = [90, 10, 260, 310]


def skeleton(hidden=(), joints=JOINTS, box=BOX):
    kp = np.array([[*joints[n], 0.0 if n in hidden else 0.9] for n in KEYPOINTS], dtype=np.float32)
    return Skeleton(kp, list(box))


@pytest.fixture
def book():
    return RuleBook.load(DEFAULT_RULES)


def test_the_arm_chain(book):
    s = skeleton()
    assert book.evaluate("direction(left_shoulder, left_elbow)", s)["value"] == "down"
    assert book.evaluate("direction(left_elbow, left_wrist)", s)["value"] == "right"
    assert book.evaluate("turn(left_shoulder, left_elbow, left_wrist)", s)["value"] == "left"  # down, then right
    assert 85 <= book.evaluate("left_elbow_angle", s)["value"] <= 95
    facts = book.evaluate_all(s)["facts"]
    assert facts["left_arm_L"] and not facts["right_arm_L"]
    assert facts["right_arm_hanging"] and not facts["left_arm_raised"]
    assert facts["facing"] == "camera" and facts["upright"] and facts["standing"]


def test_missing_joints_make_rules_unknown_not_false(book):
    s = skeleton(hidden=["left_wrist"])
    r = book.evaluate("left_elbow_angle", s)
    assert r["value"] is None and r["missing"] == ["left_wrist"]
    # an `or` already decided by a side that could be seen is not unknown
    assert book.evaluate("right_arm_hanging or left_arm_raised", s)["value"] is True
    assert book.evaluate("left_arm_raised and right_arm_raised", s)["value"] is False  # decided by the right arm
    assert book.evaluate("left_arm_raised and right_arm_hanging", s)["value"] is None
    assert book.evaluate("visible(left_wrist)", s)["value"] is False
    out = book.evaluate_all(s)
    assert "left_arm_L" in out["unknown"] and "left_arm_L" not in out["facts"]


def test_rules_are_validated(book):
    with pytest.raises(ValueError):
        book.add("bad", "__import__('os').system('true')")
    with pytest.raises(ValueError):
        book.add("bad", "left_wrist.x > 3")
    with pytest.raises(ValueError):
        book.add("bad", "no_such_joint_or_rule")
    with pytest.raises(ValueError):
        book.add("left_wrist", "True")  # a joint name cannot be redefined
    book.add("loop_a", "True")
    book.add("loop_b", "loop_a")
    book.rules["loop_a"]["expression"] = "loop_b"
    with pytest.raises(ValueError):
        book.evaluate("loop_a", skeleton())


def test_new_rules_build_on_old_ones(book):
    book.add("carrying_pose", "left_arm_L and right_arm_hanging", "one forearm out, the other arm down")
    assert book.evaluate_all(skeleton())["facts"]["carrying_pose"] is True
    lifted = dict(JOINTS, right_wrist=(116, 20), right_elbow=(118, 50))
    assert book.evaluate("carrying_pose", skeleton(joints=lifted))["value"] is False


def test_constraints_flag_implausible_skeletons(book):
    assert book.evaluate_all(skeleton())["constraints_broken"] == []
    stretched = dict(JOINTS, left_wrist=(500, 142))  # forearm five times the upper arm
    assert "left_arm_proportions" in book.evaluate_all(skeleton(joints=stretched))["constraints_broken"]


def test_signature_separates_head_from_face():
    s = skeleton()
    heads = [part_features([125, 15, 175, 70], s), part_features([126, 14, 176, 72], s)]  # holds ears
    faces = [part_features([138, 25, 162, 55], s), part_features([137, 26, 163, 56], s)]  # ears outside
    head, face = fit_signature(heads), fit_signature(faces)
    q = part_features([127, 16, 174, 69], s)
    assert signature_loglik(head, q) > signature_loglik(face, q)
    assert q["counts"]["ears"] == 2 and q["counts"]["shoulders"] == 0


def test_owner_is_the_containing_person():
    a, b = skeleton(), skeleton(box=[400, 10, 570, 310])
    assert owner([120, 20, 180, 70], [b, a]) is a
    assert owner([300, 20, 350, 70], [a, b]) is None


# ---- session tools --------------------------------------------------------------------------------
class FakePose:
    """Returns JOINTS scaled into whatever box it is given."""

    def estimate(self, image, boxes):
        out = []
        for x0, y0, x1, y1 in boxes:
            sx, sy = (x1 - x0) / (BOX[2] - BOX[0]), (y1 - y0) / (BOX[3] - BOX[1])
            out.append(np.array([[x0 + (JOINTS[n][0] - BOX[0]) * sx, y0 + (JOINTS[n][1] - BOX[1]) * sy, 0.9]
                                 for n in KEYPOINTS], dtype=np.float32))
        return out


@pytest.fixture
def geo_session(session):
    session._pose_estimator = FakePose()
    return session


def test_pose_measure_and_define_rule(geo_session, tmp_path):
    s = geo_session
    s.taught_dir = tmp_path / "taught"
    s.load_image(str(s.images_path))
    out = s.pose("img_1")
    assert len(out["people"]) == 1  # the person object; the hand and phone are not people
    person = out["people"][0]
    assert person["object_id"] is not None and person["facts"]["left_arm_L"] is True
    assert set(person["hands_estimated_at"]) == {"left_hand", "right_hand"}
    pid = person["person_id"]
    assert s.measure(pid, "direction(left_elbow, left_wrist)")["value"] == "right"
    phone = next(o for o in s.objects.values() if s.identify(o.id)["module"] == "handheld/phone")
    assert isinstance(s.measure(pid, f"touches(left_hand, {phone.id})")["value"], bool)
    r = s.define_rule("l_arm", "left_arm_L", "left forearm out")
    assert r["on_people_seen_so_far"][pid]["value"] is True
    assert (s.taught_dir / "rules.json").exists()
    assert s.pose("img_1")["people"][0]["facts"]["l_arm"] is True
    assert s.forget("l_arm")["deleted_from_disk"]


def test_teach_and_forget(geo_session, tmp_path):
    s = geo_session
    s.taught_dir = tmp_path / "taught"
    s.load_image(str(s.images_path))
    s.find_objects("img_1")
    phone = next(o for o in s.objects.values() if s.identify(o.id)["module"] == "handheld/phone")
    # a built-in module is plugged out for the session, its files stay
    out = s.forget("handheld")
    assert out["removed"] == ["handheld/phone"] and out["kept_on_disk"] == ["handheld/phone"]
    assert not s.identify(phone.id)["known"]
    # teach it back under a new name: recognised at once, saved to disk
    t = s.teach("gadget", [phone.id])
    assert t["module"] == "taught/gadget" and t["recognised_again"][phone.id] == "taught/gadget"
    assert (s.taught_dir / "taught/gadget/manifest.json").exists()
    assert s.list_modules("taught")["children"][0]["taught"] is True
    # forgetting a taught module deletes it
    out = s.forget("taught/gadget")
    assert out["deleted_from_disk"] == ["taught/gadget"]
    assert not (s.taught_dir / "taught").exists()
    assert not s.identify(phone.id)["known"]


def test_explain_shows_routing_gate_and_evidence(geo_session):
    s = geo_session
    s.load_image(str(s.images_path))
    oid = s.find_objects("img_1")["objects"][0]["object_id"]
    e = s.explain(oid)
    assert e["routing"]["levels"] and e["gate"]["module"] == e["module"]
    assert len(e["evidence"]) == 5 and all("similarity" in x for x in e["evidence"])
