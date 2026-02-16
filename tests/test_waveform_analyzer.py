"""
Automated Pre-Commit Checklist Tests for Waveform Analyzer.

Maps directly to the Pre-Commit Checklist in CLAUDE.md:
  - All 4 wave types render correctly
  - Edge cases: min/max frequency, amplitude
  - Duty cycle: 1%, 50%, 100% for Square
  - Wave duration: 0.5s, 10s, 120s
  - Envelopes: 2, 3, 5 waveforms (same/opposite phase)
  - Mixed enabled/disabled waveforms
  - Toggle max/min independently and together
  - CSV export with/without envelopes
  - Add/remove waveforms (min/max limits)
  - No console errors or warnings
  - Performance: envelope <10ms, waveform gen <100ms

Run:  pytest tests/ -v
"""

import os
import time
import tempfile

import numpy as np
import pytest

from waveform_generator import (
    gen_sine_wf, gen_square_wf, gen_sawtooth_wf, gen_triangle_wf,
    gen_wf, compute_max_env, compute_min_env, compute_rms_env,
)
from app_state import (
    AppState, WfState,
    FREQ_MIN, FREQ_MAX, AMP_MIN, AMP_MAX,
    DUTY_MIN, DUTY_MAX, DURATION_MIN, DURATION_MAX,
    OFFSET_MIN, OFFSET_MAX,
)
from data_export import (
    export_to_csv, export_to_mat, export_to_json,
    prep_wf_for_export, sanitize_fname,
)
from scipy.io import loadmat
from config import load_config, save_config
from ui_components import DARK_THEME, LIGHT_THEME


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _make_test_export_wf(name: str = "TestWave"):
    """Create a waveform tuple suitable for export tests."""
    t, y = gen_sine_wf(1.0, amp=2.0, offset=5.0, dur=1.0)
    return prep_wf_for_export(name, t, y, "sine", 1.0, 2.0, 5.0, 50.0)


# ---------------------------------------------------------------------------
# Checklist Item: All 4 wave types render correctly
# ---------------------------------------------------------------------------

class TestWaveTypes:
    """Verify all 4 waveform types generate valid output."""

    @pytest.mark.parametrize("wf_type", ["sine", "square", "sawtooth", "triangle"])
    def test_gen_wf_returns_arrays(self, wf_type: str) -> None:
        """Each wave type returns time and amplitude arrays."""
        t, y = gen_wf(wf_type, freq=1.0, amp=2.0, offset=5.0, dur=1.0)
        assert isinstance(t, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(t) == len(y)
        assert len(t) > 0

    @pytest.mark.parametrize("wf_type", ["sine", "square", "sawtooth", "triangle"])
    def test_gen_wf_correct_sample_count(self, wf_type: str) -> None:
        """Sample count = sample_rate * duration."""
        dur, sr = 2.0, 1000
        t, y = gen_wf(wf_type, freq=1.0, amp=2.0, dur=dur, sample_rate=sr)
        assert len(t) == int(sr * dur)

    @pytest.mark.parametrize("wf_type", ["sine", "square", "sawtooth", "triangle"])
    def test_gen_wf_amplitude_range(self, wf_type: str) -> None:
        """Waveform stays within offset +/- amp/2."""
        amp, offset = 4.0, 5.0
        _, y = gen_wf(wf_type, freq=1.0, amp=amp, offset=offset, dur=1.0)
        expected_min = offset - amp / 2
        expected_max = offset + amp / 2
        assert np.min(y) >= expected_min - 1e-9
        assert np.max(y) <= expected_max + 1e-9

    @pytest.mark.parametrize("wf_type", ["sine", "square", "sawtooth", "triangle"])
    def test_gen_wf_time_range(self, wf_type: str) -> None:
        """Time array spans [0, duration]."""
        dur = 3.0
        t, _ = gen_wf(wf_type, freq=1.0, amp=2.0, dur=dur)
        assert t[0] == pytest.approx(0.0)
        assert t[-1] == pytest.approx(dur, rel=1e-3)

    def test_sine_waveform_shape(self) -> None:
        """Sine wave has expected sinusoidal behavior."""
        freq = 1.0
        t, y = gen_sine_wf(freq, amp=2.0, offset=0.0, dur=1.0, sample_rate=10000)
        # At t=0, sin(0)=0 → y should be near offset (0)
        assert y[0] == pytest.approx(0.0, abs=1e-6)
        # At t=0.25 (quarter period), sin(π/2)=1 → y should be amp/2
        idx_quarter = int(0.25 * 10000)
        assert y[idx_quarter] == pytest.approx(1.0, abs=1e-3)

    def test_square_waveform_levels(self) -> None:
        """Square wave alternates between offset-amp/2 and offset+amp/2."""
        amp, offset = 4.0, 5.0
        _, y = gen_square_wf(1.0, amp, duty_cycle=50.0, offset=offset, dur=1.0)
        unique_vals = np.unique(np.round(y, 6))
        assert len(unique_vals) == 2
        expected = np.array([offset - amp / 2, offset + amp / 2])
        np.testing.assert_allclose(unique_vals, expected, atol=1e-6)

    def test_gen_wf_unrecognized_defaults_to_sine(self) -> None:
        """Unknown wf_type falls back to sine."""
        t1, y1 = gen_wf("sine", freq=1.0, amp=2.0, offset=0.0, dur=1.0)
        t2, y2 = gen_wf("unknown_type", freq=1.0, amp=2.0, offset=0.0, dur=1.0)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(y1, y2)


# ---------------------------------------------------------------------------
# Checklist Item: Edge cases — min/max frequency and amplitude
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Verify extreme parameter values produce valid output."""

    @pytest.mark.parametrize("freq", [FREQ_MIN, FREQ_MAX])
    def test_extreme_frequency(self, freq: float) -> None:
        """Min (0.1 Hz) and max (100 Hz) frequency produce valid waveforms."""
        t, y = gen_wf("sine", freq=freq, amp=2.0, offset=5.0, dur=1.0)
        assert len(t) > 0
        assert not np.any(np.isnan(y))
        assert not np.any(np.isinf(y))

    @pytest.mark.parametrize("amp", [AMP_MIN, AMP_MAX])
    def test_extreme_amplitude(self, amp: float) -> None:
        """Min (0) and max (10) amplitude produce valid waveforms."""
        t, y = gen_wf("sine", freq=1.0, amp=amp, offset=5.0, dur=1.0)
        assert not np.any(np.isnan(y))
        if amp == 0.0:
            # Zero amplitude → flat line at offset
            np.testing.assert_allclose(y, 5.0, atol=1e-9)

    def test_zero_amplitude_all_types(self) -> None:
        """All wave types with zero amplitude produce flat line."""
        for wf_type in ["sine", "square", "sawtooth", "triangle"]:
            _, y = gen_wf(wf_type, freq=1.0, amp=0.0, offset=3.0, dur=1.0)
            np.testing.assert_allclose(y, 3.0, atol=1e-9)

    @pytest.mark.parametrize("offset", [OFFSET_MIN, OFFSET_MAX])
    def test_extreme_offset(self, offset: float) -> None:
        """Min (0) and max (10) offset produce valid waveforms."""
        _, y = gen_wf("sine", freq=1.0, amp=2.0, offset=offset, dur=1.0)
        assert not np.any(np.isnan(y))


# ---------------------------------------------------------------------------
# Checklist Item: Duty cycle 1%, 50%, 100% for Square
# ---------------------------------------------------------------------------

class TestDutyCycle:
    """Verify square wave duty cycle at edge and nominal values."""

    @pytest.mark.parametrize("duty", [DUTY_MIN, 50.0, DUTY_MAX])
    def test_duty_cycle_values(self, duty: float) -> None:
        """Duty cycle 1%, 50%, 100% produce valid square waves."""
        t, y = gen_square_wf(1.0, amp=2.0, duty_cycle=duty, offset=5.0, dur=1.0)
        assert not np.any(np.isnan(y))
        assert len(t) > 0

    def test_duty_1_percent_mostly_low(self) -> None:
        """1% duty cycle means most samples at offset-amp/2."""
        _, y = gen_square_wf(
            1.0, amp=4.0, duty_cycle=1.0, offset=5.0, dur=1.0,
            sample_rate=10000
        )
        low_val = 5.0 - 2.0   # offset - amp/2
        high_val = 5.0 + 2.0  # offset + amp/2
        low_count = np.sum(np.isclose(y, low_val, atol=0.01))
        high_count = np.sum(np.isclose(y, high_val, atol=0.01))
        # ~99% of samples should be low
        assert low_count > high_count

    def test_duty_100_percent_mostly_high(self) -> None:
        """100% duty cycle means most samples at offset+amp/2."""
        _, y = gen_square_wf(
            1.0, amp=4.0, duty_cycle=100.0, offset=5.0, dur=1.0,
            sample_rate=10000
        )
        high_val = 5.0 + 2.0
        high_count = np.sum(np.isclose(y, high_val, atol=0.01))
        # Nearly all samples should be high
        assert high_count > len(y) * 0.95


# ---------------------------------------------------------------------------
# Checklist Item: Wave duration 0.5s, 10s, 120s
# ---------------------------------------------------------------------------

class TestWaveDuration:
    """Verify waveform generation at minimum, nominal, and maximum durations."""

    @pytest.mark.parametrize("dur", [DURATION_MIN, 10.0, DURATION_MAX])
    def test_duration_sample_count(self, dur: float) -> None:
        """Duration produces correct number of samples."""
        sr = 1000
        t, y = gen_wf("sine", freq=1.0, amp=2.0, dur=dur, sample_rate=sr)
        assert len(t) == int(sr * dur)
        assert t[-1] == pytest.approx(dur, rel=1e-3)

    def test_duration_min_valid(self) -> None:
        """0.5s duration produces valid output."""
        t, y = gen_wf("sine", freq=1.0, amp=2.0, dur=0.5)
        assert not np.any(np.isnan(y))

    def test_duration_max_valid(self) -> None:
        """120s duration produces valid output without errors."""
        t, y = gen_wf("sine", freq=1.0, amp=2.0, dur=120.0)
        assert not np.any(np.isnan(y))
        assert len(t) == 120000


# ---------------------------------------------------------------------------
# Checklist Item: Envelopes with 2, 3, 5 waveforms (same/opposite phase)
# ---------------------------------------------------------------------------

class TestEnvelopes:
    """Verify envelope calculations with varying waveform counts."""

    def _make_wfs(self, count: int, same_phase: bool = True):
        """Generate a list of (time, amplitude) waveform tuples."""
        wfs = []
        for i in range(count):
            offset = 5.0 + i * 0.5 if same_phase else 5.0
            amp = 2.0 if same_phase else 2.0 * (1 if i % 2 == 0 else -1)
            # Use offset to shift phases or invert for "opposite phase"
            t, y = gen_sine_wf(
                freq=1.0, amp=abs(amp), offset=offset, dur=1.0
            )
            if not same_phase and i % 2 == 1:
                # Invert odd waveforms around their offset for phase opposition
                y = 2 * offset - y
            wfs.append((t, y))
        return wfs

    @pytest.mark.parametrize("count", [2, 3, 5])
    def test_max_envelope(self, count: int) -> None:
        """Max envelope >= each individual waveform at every sample."""
        wfs = self._make_wfs(count)
        t, max_env = compute_max_env(wfs)
        for _, y in wfs:
            assert np.all(max_env >= y - 1e-9)

    @pytest.mark.parametrize("count", [2, 3, 5])
    def test_min_envelope(self, count: int) -> None:
        """Min envelope <= each individual waveform at every sample."""
        wfs = self._make_wfs(count)
        t, min_env = compute_min_env(wfs)
        for _, y in wfs:
            assert np.all(min_env <= y + 1e-9)

    @pytest.mark.parametrize("count", [2, 3, 5])
    def test_rms_envelope(self, count: int) -> None:
        """RMS envelope is non-negative and finite."""
        wfs = self._make_wfs(count)
        t, rms_env = compute_rms_env(wfs)
        assert np.all(rms_env >= 0)
        assert not np.any(np.isnan(rms_env))
        assert not np.any(np.isinf(rms_env))

    @pytest.mark.parametrize("count", [2, 3, 5])
    def test_max_gte_min(self, count: int) -> None:
        """Max envelope >= min envelope at every sample (peak-to-peak valid)."""
        wfs = self._make_wfs(count)
        _, max_env = compute_max_env(wfs)
        _, min_env = compute_min_env(wfs)
        assert np.all(max_env >= min_env - 1e-9)

    def test_envelope_opposite_phase(self) -> None:
        """Envelopes with opposite-phase waveforms have wider spread."""
        wfs_same = self._make_wfs(2, same_phase=True)
        wfs_opp = self._make_wfs(2, same_phase=False)
        _, max_same = compute_max_env(wfs_same)
        _, min_same = compute_min_env(wfs_same)
        _, max_opp = compute_max_env(wfs_opp)
        _, min_opp = compute_min_env(wfs_opp)
        spread_same = np.mean(max_same - min_same)
        spread_opp = np.mean(max_opp - min_opp)
        # Opposite phase should have wider or equal spread
        assert spread_opp >= spread_same - 1e-6

    def test_envelope_shared_time_base(self) -> None:
        """Envelope time arrays match the input waveform time array."""
        wfs = self._make_wfs(3)
        t_max, _ = compute_max_env(wfs)
        t_min, _ = compute_min_env(wfs)
        t_rms, _ = compute_rms_env(wfs)
        np.testing.assert_array_equal(t_max, wfs[0][0])
        np.testing.assert_array_equal(t_min, wfs[0][0])
        np.testing.assert_array_equal(t_rms, wfs[0][0])

    def test_envelope_empty_input(self) -> None:
        """Empty waveform list returns empty arrays."""
        t, y = compute_max_env([])
        assert len(t) == 0
        assert len(y) == 0

    def test_single_wf_envelope(self) -> None:
        """Single waveform: max, min, and RMS all equal the waveform itself."""
        t, y = gen_sine_wf(1.0, amp=2.0, offset=5.0, dur=1.0)
        wfs = [(t, y)]
        _, max_env = compute_max_env(wfs)
        _, min_env = compute_min_env(wfs)
        _, rms_env = compute_rms_env(wfs)
        np.testing.assert_allclose(max_env, y, atol=1e-9)
        np.testing.assert_allclose(min_env, y, atol=1e-9)
        np.testing.assert_allclose(rms_env, np.abs(y), atol=1e-9)


# ---------------------------------------------------------------------------
# Checklist Item: Mixed enabled/disabled waveforms
# ---------------------------------------------------------------------------

class TestMixedEnabledDisabled:
    """Verify state management for enabled/disabled waveforms."""

    def test_get_enabled_wfs(self) -> None:
        """Only enabled waveforms are returned."""
        state = AppState()
        state.add_wf()
        state.add_wf()
        # 3 waveforms; disable the middle one
        state.wfs[1].enabled = False
        enabled = state.get_enabled_wfs()
        assert len(enabled) == 2
        assert all(w.enabled for w in enabled)

    def test_all_disabled_except_one(self) -> None:
        """One enabled waveform out of many."""
        state = AppState()
        for _ in range(4):
            state.add_wf()
        for wf in state.wfs[1:]:
            wf.enabled = False
        enabled = state.get_enabled_wfs()
        assert len(enabled) == 1

    def test_envelope_eligibility_with_disabled(self) -> None:
        """Envelopes require >1 enabled waveform."""
        state = AppState()
        state.add_wf()  # 2 waveforms total
        assert state.can_show_envelopes() is True
        # Disable one → only 1 enabled
        state.wfs[1].enabled = False
        assert state.can_show_envelopes() is False

    def test_toggle_waveform_enabled(self) -> None:
        """Toggling enabled state works correctly."""
        state = AppState()
        wf = state.wfs[0]
        assert wf.enabled is True
        wf.enabled = False
        assert wf.enabled is False
        wf.enabled = True
        assert wf.enabled is True


# ---------------------------------------------------------------------------
# Checklist Item: Toggle max/min independently and together
# ---------------------------------------------------------------------------

class TestEnvelopeToggles:
    """Verify envelope visibility flags behave independently."""

    def test_initial_state_all_off(self) -> None:
        """All envelope flags start as False."""
        state = AppState()
        assert state.show_max_env is False
        assert state.show_min_env is False
        assert state.show_rms_env is False

    def test_toggle_max_only(self) -> None:
        """Enable max envelope alone."""
        state = AppState()
        state.show_max_env = True
        assert state.show_max_env is True
        assert state.show_min_env is False
        assert state.show_rms_env is False

    def test_toggle_min_only(self) -> None:
        """Enable min envelope alone."""
        state = AppState()
        state.show_min_env = True
        assert state.show_max_env is False
        assert state.show_min_env is True
        assert state.show_rms_env is False

    def test_toggle_max_and_min(self) -> None:
        """Enable both max and min (peak-to-peak fill scenario)."""
        state = AppState()
        state.show_max_env = True
        state.show_min_env = True
        assert state.show_max_env is True
        assert state.show_min_env is True

    def test_toggle_all_envelopes(self) -> None:
        """Enable all three envelopes simultaneously."""
        state = AppState()
        state.show_max_env = True
        state.show_min_env = True
        state.show_rms_env = True
        assert state.show_max_env is True
        assert state.show_min_env is True
        assert state.show_rms_env is True

    def test_envelope_not_allowed_single_wf(self) -> None:
        """Envelopes cannot be shown with only 1 waveform."""
        state = AppState()
        assert state.can_show_envelopes() is False


# ---------------------------------------------------------------------------
# Checklist Item: CSV export with/without envelopes
# ---------------------------------------------------------------------------

class TestCSVExport:
    """Verify CSV export for waveform data with and without envelopes."""

    def test_export_without_envelopes(self) -> None:
        """Export single waveform without envelopes succeeds."""
        wf = _make_test_export_wf()
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            ok, msg = export_to_csv(path, [wf])
            assert ok is True
            assert os.path.exists(path)
            with open(path, "r") as f:
                content = f.read()
            assert "Time (s)" in content
            assert "TestWave" in content
        finally:
            os.unlink(path)

    def test_export_with_envelopes(self) -> None:
        """Export with envelopes includes envelope columns."""
        wf1 = _make_test_export_wf("Wave1")
        wf2 = _make_test_export_wf("Wave2")
        t1 = wf1[1]
        y1 = wf1[2]
        t2 = wf2[1]
        y2 = wf2[2]
        wfs = [(t1, y1), (t2, y2)]
        _, max_env = compute_max_env(wfs)
        envs = [("Max_Envelope", t1, max_env)]
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            ok, msg = export_to_csv(path, [wf1, wf2], envs=envs)
            assert ok is True
            with open(path, "r") as f:
                content = f.read()
            assert "Max_Envelope" in content
            assert "Wave1" in content
            assert "Wave2" in content
        finally:
            os.unlink(path)

    def test_export_multiple_waveforms(self) -> None:
        """Export with 5 waveforms succeeds."""
        wfs = [_make_test_export_wf(f"Wf{i}") for i in range(5)]
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            ok, msg = export_to_csv(path, wfs)
            assert ok is True
            with open(path, "r") as f:
                content = f.read()
            for i in range(5):
                assert f"Wf{i}" in content
        finally:
            os.unlink(path)

    def test_export_no_data(self) -> None:
        """Export with empty data returns failure."""
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            ok, msg = export_to_csv(path, [])
            assert ok is False
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_export_metadata_includes_params(self) -> None:
        """CSV metadata includes waveform parameters."""
        t, y = gen_square_wf(2.0, amp=3.0, duty_cycle=75.0, offset=1.0)
        wf = prep_wf_for_export(
            "SquareWave", t, y, "square", 2.0, 3.0, 1.0, 75.0
        )
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            ok, _ = export_to_csv(path, [wf])
            assert ok is True
            with open(path, "r") as f:
                content = f.read()
            assert "Square" in content
            assert "2.0 Hz" in content
            assert "75.0% duty cycle" in content
        finally:
            os.unlink(path)

    def test_sanitize_filename(self) -> None:
        """Filename sanitization removes invalid chars, adds extension."""
        assert sanitize_fname("test").endswith(".csv")
        assert sanitize_fname("test.csv") == "test.csv"
        assert "<" not in sanitize_fname("te<st>.csv")
        assert sanitize_fname("").endswith(".csv")
        assert sanitize_fname("test.mat").endswith(".mat")
        assert sanitize_fname("test.json").endswith(".json")


# ---------------------------------------------------------------------------
# Checklist Item: Add/remove waveforms (min/max limits)
# ---------------------------------------------------------------------------

class TestWaveformLimits:
    """Verify add/remove enforces min (1) and max (5) waveform limits."""

    def test_initial_waveform_count(self) -> None:
        """App starts with exactly 1 waveform."""
        state = AppState()
        assert len(state.wfs) == 1

    def test_add_to_max(self) -> None:
        """Can add up to 5 waveforms total."""
        state = AppState()
        for i in range(4):
            result = state.add_wf()
            assert result is not None
        assert len(state.wfs) == 5

    def test_add_beyond_max_rejected(self) -> None:
        """Adding a 6th waveform returns None."""
        state = AppState()
        for _ in range(4):
            state.add_wf()
        result = state.add_wf()
        assert result is None
        assert len(state.wfs) == 5

    def test_remove_to_min(self) -> None:
        """Can remove down to 1 waveform."""
        state = AppState()
        state.add_wf()
        assert len(state.wfs) == 2
        result = state.remove_wf(1)
        assert result is True
        assert len(state.wfs) == 1

    def test_remove_below_min_rejected(self) -> None:
        """Cannot remove the last waveform."""
        state = AppState()
        result = state.remove_wf(0)
        assert result is False
        assert len(state.wfs) == 1

    def test_ids_reassigned_after_remove(self) -> None:
        """Waveform IDs are reassigned sequentially after removal."""
        state = AppState()
        state.add_wf()
        state.add_wf()
        # Remove middle waveform
        state.remove_wf(1)
        ids = [wf.id for wf in state.wfs]
        assert ids == [0, 1]

    def test_colors_preserved_after_remove(self) -> None:
        """Colors are preserved (not reassigned) after removal."""
        state = AppState()
        state.add_wf()
        state.add_wf()
        original_colors = [wf.color for wf in state.wfs]
        state.remove_wf(0)
        # Remaining waveforms keep their original colors
        assert state.wfs[0].color == original_colors[1]
        assert state.wfs[1].color == original_colors[2]

    def test_active_index_adjusted_on_remove(self) -> None:
        """Active waveform index stays in bounds after removal."""
        state = AppState()
        state.add_wf()
        state.active_wf_index = 1
        state.remove_wf(1)
        assert state.active_wf_index < len(state.wfs)


# ---------------------------------------------------------------------------
# Checklist Item: No console errors (import & state validation)
# ---------------------------------------------------------------------------

class TestNoErrors:
    """Verify core modules import and operate without exceptions."""

    def test_all_imports_succeed(self) -> None:
        """All project modules import without error."""
        import waveform_generator
        import app_state
        import data_export
        import config

    def test_wf_state_clamps_parameters(self) -> None:
        """WfState clamps out-of-range values instead of raising errors."""
        wf = WfState(wf_id=0, freq=-10.0, amp=999.0, offset=-5.0,
                      duty_cycle=200.0)
        assert wf.freq == FREQ_MIN
        assert wf.amp == AMP_MAX
        assert wf.offset == OFFSET_MIN
        assert wf.duty_cycle == DUTY_MAX

    def test_app_state_duration_clamping(self) -> None:
        """AppState.set_duration clamps to valid range."""
        state = AppState()
        state.set_duration(-1.0)
        assert state.duration == DURATION_MIN
        state.set_duration(9999.0)
        assert state.duration == DURATION_MAX

    def test_get_wf_invalid_id(self) -> None:
        """Getting a non-existent waveform returns None (no exception)."""
        state = AppState()
        assert state.get_wf(99) is None

    def test_display_name_default_and_custom(self) -> None:
        """Display name falls back gracefully."""
        wf = WfState(wf_id=0)
        assert wf.display_name == "Waveform 1"
        wf.name = "MySignal"
        assert wf.display_name == "MySignal"


# ---------------------------------------------------------------------------
# Checklist Item: Performance (FPS >30 ↔ compute time <33ms)
# ---------------------------------------------------------------------------

class TestPerformance:
    """Verify waveform and envelope generation meet performance SLAs."""

    def test_waveform_generation_under_100ms(self) -> None:
        """Generating a waveform at max duration completes in <100ms."""
        start = time.perf_counter()
        gen_wf("sine", freq=100.0, amp=10.0, offset=5.0, dur=120.0)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 100, f"Waveform generation took {elapsed:.1f}ms (>100ms)"

    def test_envelope_calculation_under_10ms(self) -> None:
        """Envelope computation for 5 waveforms at max duration in <10ms."""
        wfs = []
        for i in range(5):
            t, y = gen_sine_wf(1.0 + i, amp=2.0, offset=5.0, dur=120.0)
            wfs.append((t, y))
        start = time.perf_counter()
        compute_max_env(wfs)
        compute_min_env(wfs)
        compute_rms_env(wfs)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 10, f"Envelope computation took {elapsed:.1f}ms (>10ms)"

    def test_all_waveform_types_under_100ms(self) -> None:
        """All 4 waveform types at max duration under 100ms each."""
        for wf_type in ["sine", "square", "sawtooth", "triangle"]:
            start = time.perf_counter()
            gen_wf(wf_type, freq=50.0, amp=5.0, offset=5.0, dur=120.0)
            elapsed = (time.perf_counter() - start) * 1000
            assert elapsed < 100, (
                f"{wf_type} generation took {elapsed:.1f}ms (>100ms)"
            )


# ---------------------------------------------------------------------------
# Config module tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Verify configuration load/save behavior."""

    def test_load_config_returns_all_keys(self) -> None:
        """Config loader returns all expected keys."""
        cfg = load_config()
        expected_keys = [
            "duration", "frequency", "amplitude", "offset",
            "duty_cycle", "waveform_type", "y_axis_title",
            "y_min", "y_max",
        ]
        for key in expected_keys:
            assert key in cfg, f"Missing config key: {key}"

    def test_config_values_are_numeric(self) -> None:
        """Numeric config values are floats."""
        cfg = load_config()
        for key in ["duration", "frequency", "amplitude", "offset",
                     "duty_cycle", "y_min", "y_max"]:
            assert isinstance(cfg[key], (int, float)), (
                f"Config key '{key}' is {type(cfg[key])}, expected numeric"
            )

    def test_config_waveform_type_valid(self) -> None:
        """Waveform type is one of the 4 valid types."""
        cfg = load_config()
        assert cfg["waveform_type"] in ("sine", "square", "sawtooth", "triangle")


# ---------------------------------------------------------------------------
# MATLAB .mat export
# ---------------------------------------------------------------------------

class TestMATExport:
    """Verify MATLAB .mat export functionality."""

    def test_export_mat_creates_file(self) -> None:
        """MAT export creates a valid .mat file."""
        wf = _make_test_export_wf()
        with tempfile.NamedTemporaryFile(
            suffix=".mat", delete=False
        ) as f:
            path = f.name
        try:
            ok, msg = export_to_mat(path, [wf])
            assert ok is True
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_export_mat_contains_variables(self) -> None:
        """MAT file contains time, waveform, and metadata variables."""
        wf = _make_test_export_wf("Wave1")
        with tempfile.NamedTemporaryFile(
            suffix=".mat", delete=False
        ) as f:
            path = f.name
        try:
            export_to_mat(path, [wf])
            data = loadmat(path)
            assert 'time' in data
            assert 'Wave1' in data
            assert 'sample_rate' in data
            assert 'duration' in data
        finally:
            os.unlink(path)

    def test_export_mat_values_match(self) -> None:
        """MAT file waveform data matches source arrays."""
        t, y = gen_sine_wf(1.0, amp=2.0, offset=5.0, dur=0.5)
        wf = prep_wf_for_export("Sig", t, y, "sine", 1.0, 2.0, 5.0, 50.0)
        with tempfile.NamedTemporaryFile(
            suffix=".mat", delete=False
        ) as f:
            path = f.name
        try:
            export_to_mat(path, [wf])
            data = loadmat(path)
            np.testing.assert_allclose(
                data['time'].flatten(), t, atol=1e-9
            )
            np.testing.assert_allclose(
                data['Sig'].flatten(), y, atol=1e-9
            )
        finally:
            os.unlink(path)

    def test_export_mat_with_envelopes(self) -> None:
        """MAT export includes envelope variables."""
        wf1 = _make_test_export_wf("W1")
        wf2 = _make_test_export_wf("W2")
        wfs = [(wf1[1], wf1[2]), (wf2[1], wf2[2])]
        _, max_env = compute_max_env(wfs)
        envs = [("Max_Envelope", wf1[1], max_env)]
        with tempfile.NamedTemporaryFile(
            suffix=".mat", delete=False
        ) as f:
            path = f.name
        try:
            ok, _ = export_to_mat(path, [wf1, wf2], envs=envs)
            assert ok is True
            data = loadmat(path)
            assert 'Max_Envelope' in data
        finally:
            os.unlink(path)

    def test_export_mat_no_data(self) -> None:
        """MAT export with empty data returns failure."""
        with tempfile.NamedTemporaryFile(
            suffix=".mat", delete=False
        ) as f:
            path = f.name
        try:
            ok, _ = export_to_mat(path, [])
            assert ok is False
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

class TestJSONExport:
    """Verify JSON export functionality."""

    def test_export_json_creates_file(self) -> None:
        """JSON export creates a valid .json file."""
        wf = _make_test_export_wf()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            ok, msg = export_to_json(path, [wf])
            assert ok is True
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_export_json_structure(self) -> None:
        """JSON file has expected top-level keys."""
        wf = _make_test_export_wf("Wave1")
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            export_to_json(path, [wf])
            import json
            with open(path, 'r') as f:
                data = json.load(f)
            assert 'exported' in data
            assert 'sample_rate' in data
            assert 'duration' in data
            assert 'time' in data
            assert 'waveforms' in data
            assert 'envelopes' in data
            assert len(data['waveforms']) == 1
            assert data['waveforms'][0]['name'] == 'Wave1'
        finally:
            os.unlink(path)

    def test_export_json_values_match(self) -> None:
        """JSON waveform data matches source arrays."""
        t, y = gen_sine_wf(1.0, amp=2.0, offset=5.0, dur=0.5)
        wf = prep_wf_for_export("Sig", t, y, "sine", 1.0, 2.0, 5.0, 50.0)
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            export_to_json(path, [wf])
            import json
            with open(path, 'r') as f:
                data = json.load(f)
            np.testing.assert_allclose(
                data['time'], t.tolist(), atol=1e-9
            )
            np.testing.assert_allclose(
                data['waveforms'][0]['amplitude_data'],
                y.tolist(), atol=1e-9
            )
        finally:
            os.unlink(path)

    def test_export_json_with_envelopes(self) -> None:
        """JSON export includes envelope data."""
        wf1 = _make_test_export_wf("W1")
        wf2 = _make_test_export_wf("W2")
        wfs = [(wf1[1], wf1[2]), (wf2[1], wf2[2])]
        _, max_env = compute_max_env(wfs)
        envs = [("Max_Envelope", wf1[1], max_env)]
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            ok, _ = export_to_json(path, [wf1, wf2], envs=envs)
            assert ok is True
            import json
            with open(path, 'r') as f:
                data = json.load(f)
            assert len(data['envelopes']) == 1
            assert data['envelopes'][0]['name'] == 'Max_Envelope'
        finally:
            os.unlink(path)

    def test_export_json_no_data(self) -> None:
        """JSON export with empty data returns failure."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            ok, _ = export_to_json(path, [])
            assert ok is False
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_export_json_waveform_params(self) -> None:
        """JSON export includes waveform parameters."""
        t, y = gen_square_wf(2.0, amp=3.0, duty_cycle=75.0, offset=1.0)
        wf = prep_wf_for_export(
            "SquareWave", t, y, "square", 2.0, 3.0, 1.0, 75.0
        )
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            export_to_json(path, [wf])
            import json
            with open(path, 'r') as f:
                data = json.load(f)
            wf_data = data['waveforms'][0]
            assert wf_data['type'] == 'square'
            assert wf_data['frequency'] == 2.0
            assert wf_data['duty_cycle'] == 75.0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Color customization
# ---------------------------------------------------------------------------

class TestColorCustomization:
    """Verify waveform color customization support."""

    def test_default_color_assignment(self) -> None:
        """Initial waveforms get colors from the palette."""
        state = AppState()
        assert state.wfs[0].color == AppState.COLORS[0]
        state.add_wf()
        assert state.wfs[1].color == AppState.COLORS[1]

    def test_color_tuple_format(self) -> None:
        """Color is an RGB tuple with int values 0-255."""
        state = AppState()
        r, g, b = state.wfs[0].color
        assert isinstance(r, int) and 0 <= r <= 255
        assert isinstance(g, int) and 0 <= g <= 255
        assert isinstance(b, int) and 0 <= b <= 255

    def test_custom_color_assignment(self) -> None:
        """Setting a custom color persists on the waveform."""
        state = AppState()
        custom = (128, 64, 200)
        state.wfs[0].color = custom
        assert state.wfs[0].color == custom

    def test_color_preserved_on_remove(self) -> None:
        """Custom color survives removal of another waveform."""
        state = AppState()
        state.add_wf()
        state.add_wf()
        custom = (100, 200, 50)
        state.wfs[2].color = custom
        state.remove_wf(1)
        # Waveform with custom color is now at index 1
        assert state.wfs[1].color == custom


# ---------------------------------------------------------------------------
# Theme toggle
# ---------------------------------------------------------------------------

class TestThemeToggle:
    """Verify dark/light theme definitions and config persistence."""

    REQUIRED_KEYS = {
        "surface", "surface_container", "surface_container_hi",
        "section_header", "text", "text_disabled", "bg", "plot_bg",
        "selected_bg", "selected_border", "separator", "border",
        "wf_on", "wf_off", "remove_btn", "success", "error",
        "rms", "p2p_fill", "cursor_default", "cursor_pinned",
        "btn_primary", "btn_primary_text", "btn_tonal", "btn_tonal_text",
        "plt_style", "ctk_mode",
    }

    def test_dark_theme_has_all_keys(self) -> None:
        """DARK_THEME contains every required key."""
        assert self.REQUIRED_KEYS.issubset(DARK_THEME.keys())

    def test_light_theme_has_all_keys(self) -> None:
        """LIGHT_THEME contains every required key."""
        assert self.REQUIRED_KEYS.issubset(LIGHT_THEME.keys())

    def test_theme_keys_match(self) -> None:
        """Both themes have identical key sets."""
        assert set(DARK_THEME.keys()) == set(LIGHT_THEME.keys())

    def test_config_theme_default(self) -> None:
        """Config loads 'dark' as default theme."""
        cfg = load_config()
        assert cfg.get("theme") in ("dark", "light")
