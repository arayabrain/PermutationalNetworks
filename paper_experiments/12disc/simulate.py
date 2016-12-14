import numpy as np
import ode

contactgroup = ode.JointGroup()
geoms = []
objs = []

# Collision callback
def near_callback(args, geom1, geom2):
    """Callback function for the collide() method.

    This function checks if the given geoms do collide and
    creates contact joints if they do.
    """

    # Check if the objects do collide
    contacts = ode.collide(geom1, geom2)

    # Create contact joints
    world,contactgroup = args
    for c in contacts:
        #print "Col!"
        c.setBounce(0.90)
        c.setMu(0)
        j = ode.ContactJoint(world, contactgroup, c)
        j.attach(geom1.getBody(), geom2.getBody())

def setupSimulation(N):
    global world, walls, space, geoms, objs
    
    world = ode.World()
    space = ode.Space()
    world.setERP(0.8)
    world.setCFM(1E-5)
    
    objs = []
    planeConstraint = ode.Plane2DJoint(world)
    
    for i in range(N):
        ball = ode.Body(world)
        M = ode.Mass()
        M.setSphere(2500.0, 0.05)
        M.mass = 2
        ball.setMass(M)
        bx = (np.random.rand()*2-1)*0.7
        by = (np.random.rand()*2-1)*0.7
        ball.setPosition( (bx,by,0) )
        geom = ode.GeomSphere(space, radius=0.2)
        geom.setBody(ball)
        geoms.append(geom)
        planeConstraint.attach(ball, ode.environment)
        objs.append(ball)
    
    walls = []
    
    walls.append(ode.GeomPlane(space, (0,1,0), -1))
    walls.append(ode.GeomPlane(space, (1,0,0), -1))
    walls.append(ode.GeomPlane(space, (0,-1,0), -1))
    walls.append(ode.GeomPlane(space, (-1,0,0), -1))

def stepSimulation():
    space.collide((world,contactgroup), near_callback)
    world.step(2e-2)
    contactgroup.empty()


def doSimulation(N, cooldown, steps):
    setupSimulation(N)
    
    for j in range(cooldown):
        stepSimulation()
    
    for k in range(len(objs)):
		objs[k].setLinearVel((np.random.randn(), np.random.randn(), 0))
		
    seq = []
    
    for j in range(steps):
        stepSimulation()
        frame = np.zeros((4,len(objs)))
        for k in range(len(objs)):
            pos = objs[k].getPosition()
            vel = objs[k].getLinearVel()
            frame[0,k] = pos[0]
            frame[1,k] = pos[1]
            frame[2,k] = vel[0]
            frame[3,k] = vel[1]
        seq.append(frame)
    seq = np.array(seq)
    
    return seq
