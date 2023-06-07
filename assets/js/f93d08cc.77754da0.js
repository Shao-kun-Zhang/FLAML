"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[6968],{3905:(e,t,n)=>{n.d(t,{Zo:()=>p,kt:()=>g});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var l=r.createContext({}),u=function(e){var t=r.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},p=function(e){var t=u(e.components);return r.createElement(l.Provider,{value:t},e.children)},c={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,l=e.parentName,p=s(e,["components","mdxType","originalType","parentName"]),m=u(n),g=a,d=m["".concat(l,".").concat(g)]||m[g]||c[g]||o;return n?r.createElement(d,i(i({ref:t},p),{},{components:n})):r.createElement(d,i({ref:t},p))}));function g(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,i=new Array(o);i[0]=m;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s.mdxType="string"==typeof e?e:a,i[1]=s;for(var u=2;u<o;u++)i[u]=n[u];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},6337:(e,t,n)=>{n.r(t),n.d(t,{contentTitle:()=>i,default:()=>p,frontMatter:()=>o,metadata:()=>s,toc:()=>l});var r=n(7462),a=(n(7294),n(3905));const o={sidebar_label:"user_proxy_agent",title:"autogen.agent.user_proxy_agent"},i=void 0,s={unversionedId:"reference/autogen/agent/user_proxy_agent",id:"reference/autogen/agent/user_proxy_agent",isDocsHomePage:!1,title:"autogen.agent.user_proxy_agent",description:"UserProxyAgent Objects",source:"@site/docs/reference/autogen/agent/user_proxy_agent.md",sourceDirName:"reference/autogen/agent",slug:"/reference/autogen/agent/user_proxy_agent",permalink:"/FLAML/docs/reference/autogen/agent/user_proxy_agent",editUrl:"https://github.com/microsoft/FLAML/edit/main/website/docs/reference/autogen/agent/user_proxy_agent.md",tags:[],version:"current",frontMatter:{sidebar_label:"user_proxy_agent",title:"autogen.agent.user_proxy_agent"},sidebar:"referenceSideBar",previous:{title:"coding_agent",permalink:"/FLAML/docs/reference/autogen/agent/coding_agent"},next:{title:"completion",permalink:"/FLAML/docs/reference/autogen/oai/completion"}},l=[{value:"UserProxyAgent Objects",id:"userproxyagent-objects",children:[{value:"__init__",id:"__init__",children:[],level:4},{value:"auto_reply",id:"auto_reply",children:[],level:4},{value:"receive",id:"receive",children:[],level:4}],level:2}],u={toc:l};function p(e){let{components:t,...n}=e;return(0,a.kt)("wrapper",(0,r.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h2",{id:"userproxyagent-objects"},"UserProxyAgent Objects"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"class UserProxyAgent(Agent)\n")),(0,a.kt)("p",null,"(Experimental) A proxy agent for the user, that can execute code and provide feedback to the other agents."),(0,a.kt)("h4",{id:"__init__"},"_","_","init","_","_"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'def __init__(name, system_message="", work_dir=None, human_input_mode="ALWAYS", max_consecutive_auto_reply=None, is_termination_msg=None, use_docker=True, **config, ,)\n')),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"name")," ",(0,a.kt)("em",{parentName:"li"},"str")," - name of the agent"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"system_message")," ",(0,a.kt)("em",{parentName:"li"},"str")," - system message to be sent to the agent"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"work_dir")," ",(0,a.kt)("em",{parentName:"li"},"str")," - working directory for the agent"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"human_input_mode")," ",(0,a.kt)("em",{parentName:"li"},"str"),' - whether to ask for human inputs every time a message is received.\nPossible values are "ALWAYS", "TERMINATE", "NEVER".\n(1) When "ALWAYS", the agent prompts for human input every time a message is received.\nUnder this mode, the conversation stops when the human input is "exit",\nor when is_termination_msg is True and there is no human input.\n(2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or\nthe number of auto reply reaches the max_consecutive_auto_reply.\n(3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops\nwhen the number of auto reply reaches the max_consecutive_auto_reply or or when is_termination_msg is True.'),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"max_consecutive_auto_reply")," ",(0,a.kt)("em",{parentName:"li"},"int"),' - the maximum number of consecutive auto replies.\ndefault to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).\nThe limit only plays a role when human_input_mode is not "ALWAYS".'),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"is_termination_msg")," ",(0,a.kt)("em",{parentName:"li"},"function")," - a function that takes a message and returns a boolean value.\nThis function is used to determine if a received message is a termination message."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"use_docker")," ",(0,a.kt)("em",{parentName:"li"},"bool")," - whether to use docker to execute the code."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"**config")," ",(0,a.kt)("em",{parentName:"li"},"dict")," - other configurations.")),(0,a.kt)("h4",{id:"auto_reply"},"auto","_","reply"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},'def auto_reply(message, sender, default_reply="")\n')),(0,a.kt)("p",null,"Generate an auto reply."),(0,a.kt)("h4",{id:"receive"},"receive"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"def receive(message, sender)\n")),(0,a.kt)("p",null,"Receive a message from the sender agent.\nOnce a message is received, this function sends a reply to the sender or simply stop.\nThe reply can be generated automatically or entered manually by a human."))}p.isMDXComponent=!0}}]);