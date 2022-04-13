Shader "Unlit/space"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            // License CC0: Stars and galaxy
// Bit of sunday tinkering lead to stars and a galaxy
// Didn't turn out as I envisioned but it turned out to something
// that I liked so sharing it.

// Controls how many layers of stars
#define LAYERS            5.0

#define PI                3.141592654
#define TAU               (2.0*PI)
#define TIME              mod(_Time.y, 30.0)
#define TTIME             (TAU*TIME)
#define RESOLUTION        iResolution
#define ROT(a)            float2x2(cos(a), sin(a), -sin(a), cos(a))
#define mod(x,y) (x-y*floor(x/y))


// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
static float sRGB(float t) { return lerp(1.055*pow(t, 1./2.4) - 0.055, 12.92*t, step(t, 0.0031308)); }
// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
static float3 sRGB(in float3 c) { return float3 (sRGB(c.x), sRGB(c.y), sRGB(c.z)); }

// License: Unknown, author: Matt Taylor (https://github.com/64), found: https://64.github.io/tonemapping/
static float3 aces_approx(float3 v) {
  v = max(v, 0.0);
  v *= 0.6f;
  float a = 2.51f;
  float b = 0.03f;
  float c = 2.43f;
  float d = 0.59f;
  float e = 0.14f;
  return clamp((v*(a*v+b))/(v*(c*v+d)+e), 0.0f, 1.0f);
}

// License: Unknown, author: Unknown, found: don't remember
static float tanh_approx(float x) {
  //  Found this somewhere on the interwebs
  //  return tanh(x);
  float x2 = x*x;
  return clamp(x*(27.0 + x2)/(27.0+9.0*x2), -1.0, 1.0);
}


// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
static const float4 hsv2rgb_K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
static float3 hsv2rgb(float3 c) {
  float3 p = abs(frac(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www);
  return c.z * lerp(hsv2rgb_K.xxx, clamp(p - hsv2rgb_K.xxx, 0.0, 1.0), c.y);
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
static float2 mod2(inout float2 p, float2 size) {
  float2 c = floor((p + size*0.5)/size);
  p = mod(p + size*0.5,size) - size*0.5;
  return c;
}

// License: Unknown, author: Unknown, found: don't remember
static float2 hash2(float2 p) {
  p = float2(dot (p, float2 (127.1, 311.7)), dot (p, float2 (269.5, 183.3)));
  return frac(sin(p)*43758.5453123);
}

static float2 shash2(float2 p) {
  return -1.0+2.0*hash2(p);
}

static float3 toSpherical(float3 p) {
  float r   = length(p);
  float t   = acos(p.z/r);
  float ph  = atan2(p.y, p.x);
  return float3(r, t, ph);
}


// License: CC BY-NC-SA 3.0, author: Stephane Cuillerdier - Aiekick/2015 (twitter:@aiekick), found: https://www.shadertoy.com/view/Mt3GW2
static float3 blackbody(float Temp) {
  float3 col = 255.;
  col.x = 56100000. * pow(Temp,(-3. / 2.)) + 148.;
  col.y = 100.04 * log(Temp) - 623.6;
  if (Temp > 6500.) col.y = 35200000. * pow(Temp,(-3. / 2.)) + 184.;
  col.z = 194.18 * log(Temp) - 1448.6;
  col = clamp(col, 0., 255.)/255.;
  if (Temp < 1000.) col *= Temp/1000.;
  return col;
}


// License: MIT, author: Inigo Quilez, found: https://www.shadertoy.com/view/XslGRr
static float noise(float2 p) {
  // Found at https://www.shadertoy.com/view/sdlXWX
  // Which then redirected to IQ shader
  float2 i = floor(p);
  float2 f = frac(p);
  float2 u = f*f*(3.-2.*f);
  
  float n =
         lerp( lerp( dot(shash2(i + float2(0.,0.) ), f - float2(0.,0.)), 
                   dot(shash2(i + float2(1.,0.) ), f - float2(1.,0.)), u.x),
              lerp( dot(shash2(i + float2(0.,1.) ), f - float2(0.,1.)), 
                   dot(shash2(i + float2(1.,1.) ), f - float2(1.,1.)), u.x), u.y);

  return 2.0*n;              
}

static float fbm(float2 p, float o, float s, int iters) {
  p *= s;
  p += o;

  const float aa = 0.5;
  const float2x2 pp = mul(2.04, ROT(1.0));

  float h = 0.0;
  float a = 1.0;
  float d = 0.0;
  for (int i = 0; i < iters; ++i) {
    d += a;
    h += a*noise(p);
    p += float2(10.7, 8.3);
    p *= mul(pp,p);
    a *= aa;
  }
  h /= d;
  
  return h;
}

static float height(float2 p) {
  float h = fbm(p, 0.0, 5.0, 5);
  h *= 0.3;
  h += 0.0;
  return (h);
}

static float3 stars(float3 ro, float3 rd, float2 sp, float hh) {
  float3 col = 0.0;
  
  const float m = LAYERS;
  hh = tanh_approx(20.0*hh);

  for (float i = 0.0; i < m; ++i) {
    float2 pp = sp+0.5*i;
    float s = i/(m-1.0);
    float2 dim  = lerp(0.05, 0.003, s)*PI;
    float2 np = mod2(pp, dim);
    float2 h = hash2(np+127.0+i);
    float2 o = -1.0+2.0*h;
    float y = sin(sp.x);
    pp += o*dim*0.5;
    pp.y *= y;
    float l = length(pp);
  
    float h1 = frac(h.x*1667.0);
    float h2 = frac(h.x*1887.0);
    float h3 = frac(h.x*2997.0);

    float3 scol = lerp(8.0*h2, 0.25*h2*h2, s)*blackbody(lerp(3000.0, 22000.0, h1*h1));

    float3 ccol = col + exp(-(lerp(6000.0, 2000.0, hh)/lerp(2.0, 0.25, s))*max(l-0.001, 0.0))*scol;
    col = h3 < y ? ccol : col;
  }
  
  return col;
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/spherefunctions/spherefunctions.htm
static float2 raySphere(float3 ro, float3 rd, float4 sph) {
  float3 oc = ro - sph.xyz;
  float b = dot( oc, rd );
  float c = dot( oc, oc ) - sph.w*sph.w;
  float h = b*b - c;
  if( h<0.0 ) return -1.0;
  h = sqrt( h );
  return float2(-b - h, -b + h);
}


static float4 moon(float3 ro, float3 rd, float2 sp, float3 lp, float4 md) {
  float2 mi = raySphere(ro, rd, md);
  
  float3 p    = ro + mi.x*rd;
  float3 n    = normalize(p-md.xyz);
  float3 r    = reflect(rd, n);
  float3 ld   = normalize(lp - p);
  float fre = dot(n, rd)+1.0;
  fre = pow(fre, 15.0);
  float dif = max(dot(ld, n), 0.0);
  float spe = pow(max(dot(ld, r), 0.0), 8.0);
  float i = 0.5*tanh_approx(20.0*fre*spe+0.05*dif);
  float3 col = blackbody(1500.0)*i+hsv2rgb(float3(0.6, lerp(0.6, 0.0, i), i));

  float t = tanh_approx(0.25*(mi.y-mi.x));
 
  return float4(float3(col), t);
}

static float3 sky(float3 ro, float3 rd, float2 sp, float3 lp, out float cf) {
  float ld = max(dot(normalize(lp-ro), rd),0.0);
  float y = -0.5+sp.x/PI;
  y = max(abs(y)-0.02, 0.0)+0.1*smoothstep(0.5, PI, abs(sp.y));
  float3 blue = hsv2rgb(float3(0.6, 0.75, 0.35*exp(-15.0*y)));
  float ci = pow(ld, 10.0)*2.0*exp(-25.0*y); 
  float3 yellow = blackbody(1500.0)*ci;
  cf = ci;
  return blue+yellow;
}

static float3 galaxy(float3 ro, float3 rd, float2 sp, out float sf) {
  float2 gp = sp;
  gp *= 0.67; // NOT SHURE
  gp += float2(-1.0, 0.5);
  float h1 = height(2.0*sp);
  float gcc = dot(gp, gp);
  float gcx = exp(-(abs(3.0*(gp.x))));
  float gcy = exp(-abs(10.0*(gp.y)));
  float gh = gcy*gcx;
  float cf = smoothstep(0.05, -0.2, -h1);
  float3 col = 0.0;
  col += blackbody(lerp(300.0, 1500.0, gcx*gcy))*gcy*gcx;
  col += hsv2rgb(float3(0.6, 0.5, 0.00125/gcc));
  col *= lerp(lerp(0.15, 1.0, gcy*gcx), 1.0, cf);
  sf = gh*cf;
  return col;
}

float3 grid(float3 ro, float3 rd, float2 sp) {
  const float m = 1.0;

  const float2 dim = 1.0/8.0*PI;
  float2 pp = sp;
  float2 np = mod2(pp, dim);

  float3 col = 0.0;

  float y = sin(sp.x);
  float d = min(abs(pp.x), abs(pp.y*y));
  

  
  col += 2.0*float3(0.5, 0.5, 1.0)*exp(-2000.0*max(d-0.00025, 0.0));
  
  return 0.25*tanh(col);
}

float3 color(float3 ro, float3 rd, float3 lp, float4 md) {
  float2 sp = toSpherical(rd.xzy).yz;

  float sf = 0.0;
  float cf = 0.0;
  float3 col = 0.0;

  float4 mcol = moon(ro, rd, sp, lp, md);

  col += stars(ro, rd, sp, sf)*(1.0-tanh_approx(2.0*cf));
  col += galaxy(ro, rd, sp, sf);
  col = lerp(col, mcol.xyz, mcol.w);
  col += sky(ro, rd, sp, lp, cf);
  col += grid(ro, rd, sp);

  if (rd.y < 0.0)
  {
    col = 0.0;
  }

  return col;
}

            fixed4 frag (v2f i) : SV_Target
            {
  float2 q = i.uv;
  float2 p = -1.0 + 2.0*q;


  float3 ro = float3(0.0, 0.0, 0.0);
  float3 lp = 500.0*float3(1.0, -0.25, 0.0);
  float4 md = 50.0*float4(float3(1.0, 1., -0.6), 0.5);
  float3 la = float3(1.0, 0.5, 0.0);
  float3 up = float3(0.0, 1.0, 0.0);
  la.xz *= TTIME/60.0-PI/2.0;
  
  float3 ww = normalize(la - ro);
  float3 uu = normalize(cross(up, ww));
  float3 vv = normalize(cross(ww,uu));
  float3 rd = normalize(p.x*uu + p.y*vv + 2.0*ww);
  float3 col= color(ro, rd, lp, md);
  
  col *= smoothstep(0.0, 4.0, TIME)*smoothstep(30.0, 26.0, TIME);
  col = aces_approx(col);
  col = sRGB(col);

  return float4(col,1.0);
}




            ENDCG
        }
    }
}
