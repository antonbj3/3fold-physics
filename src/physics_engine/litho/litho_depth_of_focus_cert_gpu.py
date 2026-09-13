"""GPU defocus propagation with the CPU sibling's unchanged four own gates."""
import types
from physics_engine.litho import litho_depth_of_focus_cert as cpu
from physics_engine.litho.litho_aerial_image_scalar_gpu import aerial_image


def aerial_contrast_defocus(k1,NA,lam,sigma,defocus,Nper=24,npts=8192,device='cuda'):
    image=aerial_image(k1,NA,lam,sigma,defocus,Nper,npts,device=device)
    hi,lo=image.max(),image.min()
    return float(((hi-lo)/(hi+lo+1e-30)).item())


def main(device='cuda'):
    scope=dict(vars(cpu))
    scope['aerial_contrast_defocus']=lambda *a,**kw:aerial_contrast_defocus(*a,**kw,device=device)
    return types.FunctionType(cpu.main.__code__,scope,cpu.main.__name__)()

if __name__=='__main__':raise SystemExit(main())
