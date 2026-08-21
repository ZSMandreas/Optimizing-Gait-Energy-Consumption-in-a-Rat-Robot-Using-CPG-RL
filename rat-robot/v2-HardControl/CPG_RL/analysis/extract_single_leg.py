"""
Extract full single-leg (FL) physical parameters from MuJoCo model.
No simplification; values are read directly from model/runtime.
"""

import os
import numpy as np
import mujoco

from rat_cpg_env_energy_substep50 import RatCPGEnv, LegModel


def main() -> None:
    model_path = (
        "/home/user/rat_ws/Optimizing-Gait-Energy-Consumption-in-a-Rat-Robot-Using-CPG-RL/"
        "rat-robot/v2-HardControl/TrotGait/models/dynamic_4l.xml"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    env = RatCPGEnv(model_path=model_path)
    mujoco.mj_forward(model, data)

    print("=" * 60)
    print("Single-leg parameter extraction (FL: thigh + leg)")
    print("=" * 60)

    # 1) Joint info
    fl_joints = ["thigh_joint_fl", "leg_joint_fl"]
    print("\n[Joint info]")
    joint_info = {}
    for jname in fl_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dof = int(model.jnt_dofadr[jid])
        qpos = int(model.jnt_qposadr[jid])
        rng = model.jnt_range[jid]
        axis = model.jnt_axis[jid]
        print(f"  {jname}:")
        print(f"    joint_id={jid}, dof={dof}, qpos={qpos}")
        print(f"    range=[{np.degrees(rng[0]):.1f} deg, {np.degrees(rng[1]):.1f} deg]")
        print(f"    axis={axis}")
        joint_info[jname] = {"id": jid, "dof": dof, "qpos": qpos, "range": rng}

    # 2) Actuator info
    fl_acts = ["thigh_joint_fl", "leg_joint_fl"]
    print("\n[Actuator info]")
    act_info = {}
    for aname in fl_acts:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        kp = float(model.actuator_gainprm[aid, 0])
        fr = model.actuator_forcerange[aid]
        gear = float(model.actuator_gear[aid, 0])
        print(f"  {aname}:")
        print(f"    actuator_id={aid}, kp={kp}, gear={gear}")
        print(f"    forcerange=[{fr[0]:.6f}, {fr[1]:.6f}] N*m")
        act_info[aname] = {"id": aid, "kp": kp, "tau_max": abs(fr[1]), "gear": gear}

    # 3) Body masses/inertia for FL side
    print("\n[FL-side body mass and inertia]")
    fl_body_keywords = ["fl", "front_left", "FL"]
    leg_bodies = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and any(k.lower() in name.lower() for k in fl_body_keywords):
            mass = float(model.body_mass[i])
            inertia = model.body_inertia[i]
            pos = model.body_pos[i]
            if mass > 1e-8:
                leg_bodies[name] = {"mass": mass, "inertia": inertia, "pos": pos}
                print(f"  {name}:")
                print(f"    mass={mass:.8f} kg")
                print(
                    f"    inertia=[{inertia[0]:.10f}, {inertia[1]:.10f}, {inertia[2]:.10f}] kg*m^2"
                )
                print(f"    rel_pos={pos}")
    m_leg_total = sum(x["mass"] for x in leg_bodies.values())
    print(f"  FL total mass={m_leg_total:.8f} kg")

    # 4) Effective inertia J_eff under several postures
    print("\n[Effective inertia J_eff at different postures]")
    nv = model.nv
    thigh_dof = joint_info["thigh_joint_fl"]["dof"]
    leg_dof = joint_info["leg_joint_fl"]["dof"]
    thigh_qpos = joint_info["thigh_joint_fl"]["qpos"]
    leg_qpos = joint_info["leg_joint_fl"]["qpos"]

    postures = {
        "qpos0": None,
        "straight(thigh=0,leg=0)": (0.0, 0.0),
        "standing(thigh=-0.3,leg=0.5)": (-0.3, 0.5),
        "mid-swing(thigh=0.5,leg=-0.5)": (0.5, -0.5),
    }

    j_eff_results = {}
    for posture_name, angles in postures.items():
        data2 = mujoco.MjData(model)
        data2.qpos[:] = model.qpos0
        if angles is not None:
            data2.qpos[thigh_qpos] = angles[0]
            data2.qpos[leg_qpos] = angles[1]
        mujoco.mj_forward(model, data2)
        m_full = np.zeros((nv, nv))
        mujoco.mj_fullM(model, m_full, data2.qM)
        j_th = float(m_full[thigh_dof, thigh_dof])
        j_lg = float(m_full[leg_dof, leg_dof])
        j_cross = float(m_full[thigh_dof, leg_dof])
        j_eff_results[posture_name] = {
            "J_thigh": j_th,
            "J_leg": j_lg,
            "J_cross": j_cross,
        }
        print(f"  {posture_name}:")
        print(f"    J_thigh={j_th:.10f} kg*m^2")
        print(f"    J_leg={j_lg:.10f} kg*m^2")
        print(f"    J_cross={j_cross:.10f} kg*m^2")

    # 5) LegModel IK test
    print("\n[IK parameters and test]")
    leg_params = [0.031, 0.0128, 0.0118, 0.040, 0.015, 0.035]
    print(f"  leg_params={leg_params}")
    lm = LegModel(leg_params)
    test_pts = [
        (-0.00, -0.045, "center/stance"),
        (-0.00, -0.037, "front peak"),
        (0.02, -0.042, "forward swing"),
        (-0.02, -0.042, "backward swing"),
    ]
    for fy, fz, desc in test_pts:
        q = lm.pos_2_angle(fy, fz)
        print(f"  ({fy:.3f}, {fz:.3f}) [{desc}] -> q={q}")

    # 6) Foot site position
    print("\n[Foot site position]")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ankle_fl")
    if site_id >= 0:
        mujoco.mj_forward(model, data)
        print(f"  ankle_fl xyz at qpos0: {data.site_xpos[site_id]}")

    # 7) Default CPG params
    print("\n[Current default CPG params]")
    print("  FU: y0=-0.00, z0=-0.045, a=0.03, b=0.01")
    print("  FD: y0=-0.00, z0=-0.045, a=0.03, b=0.005")
    print("  HU: y0=-0.005, z0=-0.05, a=0.03, b=0.01")
    print("  HD: y0=-0.005, z0=-0.05, a=0.03, b=0.005")
    print("  trot phase: [FL=0, FR=pi, RL=pi, RR=0]")

    # 8) Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  kp(thigh)={act_info['thigh_joint_fl']['kp']}")
    print(f"  tau_max(thigh)={act_info['thigh_joint_fl']['tau_max']:.6f} N*m")
    print(
        "  J_thigh range=[{:.10f}, {:.10f}] kg*m^2".format(
            min(r["J_thigh"] for r in j_eff_results.values()),
            max(r["J_thigh"] for r in j_eff_results.values()),
        )
    )
    print(f"  m_leg_total={m_leg_total:.8f} kg")
    print(f"  leg_params={leg_params}")

    env.close()


if __name__ == "__main__":
    main()
