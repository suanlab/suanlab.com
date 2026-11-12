"""Independent two-spring perturbation and long-time checks, standard library only."""
import json
import math
import sys


def matrix_vector(a, x):
    return [sum(r[j]*x[j] for j in range(2)) for r in a]


def symmetric_norm(a):
    center = (a[0][0]+a[1][1])/2
    radius = math.hypot((a[0][0]-a[1][1])/2, a[0][1])
    return max(abs(center+radius), abs(center-radius))


def modal_family(epsilon, alpha, m=1, omega=1):
    co, sn = math.cos(2*alpha), math.sin(2*alpha)
    b = [[co,sn],[sn,-co]]
    d = [[omega*omega*((1 if i==j else 0)+epsilon*b[i][j]) for j in range(2)] for i in range(2)]
    projector = [[((1 if i==j else 0)+b[i][j])/2 for j in range(2)] for i in range(2)]
    coupling = -m*omega*omega*epsilon*sn
    anchors = [m*omega*omega*(1+epsilon*co+epsilon*sn), m*omega*omega*(1-epsilon*co+epsilon*sn)]
    return d, projector, anchors, coupling


def family_state(epsilon, alpha, omega, amplitude, t):
    high, low = omega*math.sqrt(1+epsilon), omega*math.sqrt(1-epsilon)
    co, sn = math.cos(alpha), math.sin(alpha)
    def combine(fhigh, flow):
        return [amplitude*(co*co*fhigh+sn*sn*flow), amplitude*co*sn*(fhigh-flow)]
    q = combine(math.cos(high*t), math.cos(low*t))
    v = combine(-high*math.sin(high*t), -low*math.sin(low*t))
    a = combine(-high*high*math.cos(high*t), -low*low*math.cos(low*t))
    return q, v, a


def beat_state(m, k, epsilon, amplitude, t):
    omega = math.sqrt(k/m)
    high, approximate = omega*math.sqrt(1+2*epsilon), omega*(1+epsilon)
    delta = omega*epsilon*epsilon/(1+epsilon+math.sqrt(1+2*epsilon))
    def state(w):
        return ([amplitude*(math.cos(omega*t)+math.cos(w*t))/2, amplitude*(math.cos(omega*t)-math.cos(w*t))/2],
                [-amplitude*(omega*math.sin(omega*t)+w*math.sin(w*t))/2, -amplitude*(omega*math.sin(omega*t)-w*math.sin(w*t))/2])
    return state(high), state(approximate), high, approximate, delta


def energy(m, k, coupling, q, v):
    return m*sum(x*x for x in v)/2+k*sum(x*x for x in q)/2+coupling*(q[0]-q[1])**2/2


def rk4(d, amplitude, duration, steps):
    h = duration/steps
    y = [amplitude,0,0,0]
    def rhs(z):
        acceleration = matrix_vector(d,z[:2])
        return [z[2],z[3],-acceleration[0],-acceleration[1]]
    for _ in range(steps):
        a=rhs(y);b=rhs([y[i]+h*a[i]/2 for i in range(4)])
        c=rhs([y[i]+h*b[i]/2 for i in range(4)]);f=rhs([y[i]+h*c[i] for i in range(4)])
        y=[y[i]+h*(a[i]+2*b[i]+2*c[i]+f[i])/6 for i in range(4)]
    return y


def verify():
    n = dict(physicalSpringFamilies=0, modalOdeStates=0, finiteTimeBounds=0, projectorGapPairs=0, rk4Refinements=0, degenerateBasisStates=0, nearbyModelStates=0, longHorizonStates=0)
    for m in [.5,1,2]:
        for omega in [.5,1,2]:
            for e in [.001,.0625,.4]:
                base_d, base_p, _, _ = modal_family(e,0,m,omega)
                for alpha in [-math.pi/2,-math.pi/3,-math.pi/4,-math.pi/8,0]:
                    d,p,anchors,coupling = modal_family(e,alpha,m,omega)
                    assert min(anchors)>0 and coupling>=-1e-12
                    k = [[anchors[0]+coupling,-coupling],[-coupling,anchors[1]+coupling]]
                    assert max(abs(k[i][j]/m-d[i][j]) for i in range(2) for j in range(2))<1e-12
                    for i in range(2):
                        for j in range(2):
                            assert abs(sum(p[i][a]*p[a][j] for a in range(2))-p[i][j])<1e-12
                    for lam,u in [(omega*omega*(1+e),[math.cos(alpha),math.sin(alpha)]),(omega*omega*(1-e),[-math.sin(alpha),math.cos(alpha)])]:
                        assert max(abs(x-lam*y) for x,y in zip(matrix_vector(d,u),u))<1e-12
                    n['physicalSpringFamilies']+=1
                    initial_modal=m*omega*omega*((1+e)*math.cos(alpha)**2+(1-e)*math.sin(alpha)**2)/2
                    initial_physical=m*omega*omega*(1+e*math.cos(2*alpha))/2
                    assert abs(initial_modal-initial_physical)<1e-12
                    difference_d=[[d[i][j]-base_d[i][j] for j in range(2)] for i in range(2)]
                    difference_p=[[p[i][j]-base_p[i][j] for j in range(2)] for i in range(2)]
                    pd,dd=symmetric_norm(difference_p),symmetric_norm(difference_d)
                    assert abs(pd-abs(math.sin(alpha)))<1e-12
                    assert abs(dd-2*e*omega*omega*pd)<1e-12
                    n['projectorGapPairs']+=1
                    for t in [0,.3/omega,1/omega,5/omega]:
                        q,v,a=family_state(e,alpha,omega,1,t)
                        assert max(abs(x+y) for x,y in zip(a,matrix_vector(d,q)))<2e-12
                        physical_energy=m*sum(z*z for z in v)/2+sum(anchors[i]*q[i]**2 for i in range(2))/2+coupling*(q[0]-q[1])**2/2
                        assert abs(physical_energy-initial_physical)<2e-11
                        bound=omega*t*e/(1+math.sqrt(1-e))
                        assert math.hypot(q[0]-math.cos(omega*t),q[1])<=bound+1e-12
                        n['modalOdeStates']+=1;n['finiteTimeBounds']+=1
                    if m==1 and omega==1:
                        expected=sum(family_state(e,alpha,omega,1,3)[:2],[])
                        coarse,fine=rk4(d,1,3,600),rk4(d,1,3,1200)
                        assert max(abs(x-y) for x,y in zip(coarse,fine))<2e-9
                        assert max(abs(x-y) for x,y in zip(fine,expected))<2e-10
                        n['rk4Refinements']+=1
            for alpha in [-math.pi/2,-math.pi/4,0]:
                for t in [0,1/omega,5/omega]:
                    q,v,_=family_state(0,alpha,omega,1,t)
                    assert math.hypot(q[0]-math.cos(omega*t),q[1])<1e-12
                    assert abs(energy(m,m*omega*omega,0,q,v)-m*omega*omega/2)<1e-12
                    n['degenerateBasisStates']+=1
    for m in [.5,1,2]:
        for k in [.5,1,2]:
            omega=math.sqrt(k/m)
            for e in [.01,.0625,.25,.5]:
                for amplitude in [.5,1.5]:
                    for t in [0,.2/omega,1/omega,7/omega,20/omega]:
                        exact,approx,high,wa,delta=beat_state(m,k,e,amplitude,t)
                        q,v=approx;coupling=k*e;change=k*e*e/2
                        original=(k+coupling)*amplitude*amplitude/2
                        modified=(k+coupling+change)*amplitude*amplitude/2
                        assert abs(energy(m,k,coupling+change,q,v)-modified)<1e-11
                        drift=energy(m,k,coupling,q,v)-original
                        expected=k*amplitude*amplitude*e*e/4*math.sin(wa*t)**2
                        assert abs(drift-expected)<1e-11
                        acc=[-amplitude*(omega*omega*math.cos(omega*t)+wa*wa*math.cos(wa*t))/2,-amplitude*(omega*omega*math.cos(omega*t)-wa*wa*math.cos(wa*t))/2]
                        residual=[m*acc[0]+k*q[0]+coupling*(q[0]-q[1]),m*acc[1]+k*q[1]+coupling*(q[1]-q[0])]
                        expected_r=[-change*amplitude*math.cos(wa*t),change*amplitude*math.cos(wa*t)]
                        assert max(abs(x-y) for x,y in zip(residual,expected_r))<1e-11
                        norm=math.sqrt(sum((q[i]-exact[0][i])**2+(v[i]-exact[1][i])**2/(high*high) for i in range(2)))
                        unit=math.sqrt(2)*amplitude*abs(math.sin(delta*t/2))
                        assert abs(norm-unit)<=amplitude/math.sqrt(2)*delta/high+1e-11
                        assert norm<=amplitude/math.sqrt(2)*delta*(t+1/high)+1e-11
                        n['nearbyModelStates']+=1
            for e in [.25,.125,.0625,.03125,.015625,.0078125]:
                t=2/(omega*e*e)
                exact,approx,high,wa,delta=beat_state(m,k,e,1,t)
                norm=math.sqrt(sum((approx[0][i]-exact[0][i])**2+(approx[1][i]-exact[1][i])**2/(high*high) for i in range(2)))
                assert abs(norm-math.sqrt(2)*abs(math.sin(delta*t/2)))<=delta/(math.sqrt(2)*high)+3e-11
                assert .8<delta*t<1 and .5<norm<.7
                n['longHorizonStates']+=1
    return n


if __name__=='__main__':
    result=verify()
    print(json.dumps(result,sort_keys=True) if '--json' in sys.argv else 'Coupled research checks passed: '+json.dumps(result,sort_keys=True))
