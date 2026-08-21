from mujoco_py import load_model_from_path, MjSim, MjViewer

DEAD_TIME = 10000
if __name__ == '__main__':
	#model = load_model_from_path("../Model/fore_right_leg.xml")
	model = load_model_from_path("../Model/fore_left_leg.xml")
	#model = load_model_from_path("../Model_v1/fore_right_leg.xml")
	sim = MjSim(model)
	viewer = MjViewer(sim)

	viewer.cam.lookat[0] += 0.1
	viewer.cam.lookat[1] += 0
	viewer.cam.lookat[2] += 0
	viewer.cam.elevation = 0
	viewer.cam.azimuth = 180
	viewer.cam.distance = model.stat.extent*0.05

	for i in range(DEAD_TIME):
		#sim.data.ctrl[:] = [0.5,1.35]#[-0.5096361376616533, 0.7007000675860047]#[0,1]
		sim.data.ctrl[:] = [0,0]
		sim.step()
		viewer.render()
		#originPoint = sim.data.get_site_xpos('thigh_up_point_1')
		#currentPoint = sim.data.get_site_xpos('thigh_up_point_2')
		#print(originPoint, currentPoint)
		#originPoint = sim.data.get_site_xpos('leg_link')
		#currentPoint = sim.data.get_site_xpos('ankle')
		#tY = currentPoint[1]-originPoint[1]
		#tZ = currentPoint[2]-originPoint[2]
		#print(tY, tZ)