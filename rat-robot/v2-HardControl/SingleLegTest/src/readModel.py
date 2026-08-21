from mujoco_py import load_model_from_path, MjSim, MjViewer

DEAD_TIME = 5000
if __name__ == '__main__':
	model = load_model_from_path("../Model/fore_right_leg.xml")
	sim = MjSim(model)
	viewer = MjViewer(sim)

	for i in range(DEAD_TIME):
		sim.data.ctrl[:] = [-0.5096361376616533, 0.7007000675860047]#[0,1]
		sim.step()
		viewer.render()

		originPoint = sim.data.get_site_xpos('leg_link')
		currentPoint = sim.data.get_site_xpos('ankle')
		tY = currentPoint[1]-originPoint[1]
		tZ = currentPoint[2]-originPoint[2]
		print(tY, tZ)