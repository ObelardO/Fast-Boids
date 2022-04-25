Shader "TC/InstancedBoids"
{
    SubShader
    {
        Tags
        { 
            "RenderType" = "Opaque"
            "LightMode"="ForwardBase"
        }

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing

            #include "UnityCG.cginc"
            #include "UnityLightingCommon.cginc" // for _LightColor0


            struct appdata_t
            {
                float4 vertex   : POSITION;
                float4 color    : COLOR;
            };

            struct v2f
            {
                UNITY_VERTEX_INPUT_INSTANCE_ID
                float4 vertex   : SV_POSITION;
                fixed4 color : COLOR;
                fixed4 diff : COLOR1; // diffuse lighting color
            };

            struct MeshProperties
            {
                float4x4 mat;
                float4 color;
            };

            StructuredBuffer<MeshProperties> _Properties;

            v2f vert(appdata_base i, uint instanceID: SV_InstanceID)
            {
                v2f o;

                float4 pos = mul(_Properties[instanceID].mat, i.vertex);
                o.vertex = UnityObjectToClipPos(pos);

                // get vertex normal in world space
                //half3 worldNormal = UnityObjectToWorldNormal(i.normal);
                half3 norm = mul(_Properties[instanceID].mat, i.normal);
                half3 worldNormal = UnityObjectToWorldNormal(norm);

                // dot product between normal and light direction for
                // standard diffuse (Lambert) lighting
                half nl = max(0, dot(worldNormal, _WorldSpaceLightPos0.xyz));

                // factor in the light color
                o.diff = nl * _LightColor0;

                // add illumination from ambient or light probes
                o.diff.rgb += ShadeSH9(half4(worldNormal, 1));

                o.color = _Properties[instanceID].color;

                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                return i.color * i.diff;
            }

            ENDCG
        }
    }
}