

def test_load_npz_sources_skips_foreign_layout_and_prefers_a_face(tmp_path):
    import numpy as np
    from src.eval import analysis as A

    emb = np.ones((2, 4), np.float32)
    np.savez(tmp_path / "a.npz", ids=np.array(["x", "y"]), emb=emb, ok=np.array([False, True]))
    np.savez(tmp_path / "b.npz", ids=np.array(["x", "y"]), emb=emb * 2, ok=np.array([True, True]))
    np.savez(tmp_path / "s03.npz", ids=np.array(["z"]), emb=emb[:1], det_score=np.array([0.9]), n_faces=np.array([1]))
    out = A.load_npz_sources([tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "s03.npz"])
    assert set(out) == {"x", "y"}
    assert out["x"] is not None and float(out["x"][0]) == 2.0      # a later file with a face replaces a missing one
    assert float(out["y"][0]) == 1.0
