"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[708],{3905:(e,t,n)=>{n.d(t,{Zo:()=>u,kt:()=>d});var a=n(7294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},o=Object.keys(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var s=a.createContext({}),c=function(e){var t=a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},u=function(e){var t=c(e.components);return a.createElement(s.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},m=a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,o=e.originalType,s=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),m=c(n),d=r,g=m["".concat(s,".").concat(d)]||m[d]||p[d]||o;return n?a.createElement(g,i(i({ref:t},u),{},{components:n})):a.createElement(g,i({ref:t},u))}));function d(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var o=n.length,i=new Array(o);i[0]=m;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:r,i[1]=l;for(var c=2;c<o;c++)i[c]=n[c];return a.createElement.apply(null,i)}return a.createElement.apply(null,n)}m.displayName="MDXCreateElement"},3642:(e,t,n)=>{n.r(t),n.d(t,{contentTitle:()=>i,default:()=>u,frontMatter:()=>o,metadata:()=>l,toc:()=>s});var a=n(7462),r=(n(7294),n(3905));const o={sidebar_label:"user_proxy_agent",title:"autogen.agentchat.user_proxy_agent"},i=void 0,l={unversionedId:"reference/autogen/agentchat/user_proxy_agent",id:"reference/autogen/agentchat/user_proxy_agent",isDocsHomePage:!1,title:"autogen.agentchat.user_proxy_agent",description:"UserProxyAgent Objects",source:"@site/docs/reference/autogen/agentchat/user_proxy_agent.md",sourceDirName:"reference/autogen/agentchat",slug:"/reference/autogen/agentchat/user_proxy_agent",permalink:"/FLAML/docs/reference/autogen/agentchat/user_proxy_agent",editUrl:"https://github.com/microsoft/FLAML/edit/main/website/docs/reference/autogen/agentchat/user_proxy_agent.md",tags:[],version:"current",frontMatter:{sidebar_label:"user_proxy_agent",title:"autogen.agentchat.user_proxy_agent"},sidebar:"referenceSideBar",previous:{title:"responsive_agent",permalink:"/FLAML/docs/reference/autogen/agentchat/responsive_agent"},next:{title:"completion",permalink:"/FLAML/docs/reference/autogen/oai/completion"}},s=[{value:"UserProxyAgent Objects",id:"userproxyagent-objects",children:[{value:"__init__",id:"__init__",children:[],level:4}],level:2}],c={toc:s};function u(e){let{components:t,...n}=e;return(0,r.kt)("wrapper",(0,a.Z)({},c,n,{components:t,mdxType:"MDXLayout"}),(0,r.kt)("h2",{id:"userproxyagent-objects"},"UserProxyAgent Objects"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"class UserProxyAgent(ResponsiveAgent)\n")),(0,r.kt)("p",null,"(In preview) A proxy agent for the user, that can execute code and provide feedback to the other agents."),(0,r.kt)("p",null,"UserProxyAgent is a subclass of ResponsiveAgent configured with ",(0,r.kt)("inlineCode",{parentName:"p"},"human_input_mode")," to ALWAYS\nand ",(0,r.kt)("inlineCode",{parentName:"p"},"llm_config")," to False. By default, the agent will prompt for human input every time a message is received.\nCode execution is enabled by default. LLM-based auto reply is disabled by default.\nTo modify auto reply, register a method with (",(0,r.kt)("inlineCode",{parentName:"p"},"register_auto_reply"),")","[responsive_agent#register_auto_reply]",".\nTo modify the way to get human input, override ",(0,r.kt)("inlineCode",{parentName:"p"},"get_human_input")," method.\nTo modify the way to execute code blocks, single code block, or function call, override ",(0,r.kt)("inlineCode",{parentName:"p"},"execute_code_blocks"),",\n",(0,r.kt)("inlineCode",{parentName:"p"},"run_code"),", and ",(0,r.kt)("inlineCode",{parentName:"p"},"execute_function")," methods respectively.\nTo customize the initial message when a conversation starts, override ",(0,r.kt)("inlineCode",{parentName:"p"},"generate_init_message")," method."),(0,r.kt)("h4",{id:"__init__"},"_","_","init","_","_"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},'def __init__(name: str, is_termination_msg: Optional[Callable[[Dict], bool]] = None, max_consecutive_auto_reply: Optional[int] = None, human_input_mode: Optional[str] = "ALWAYS", function_map: Optional[Dict[str, Callable]] = None, code_execution_config: Optional[Union[Dict, bool]] = None, default_auto_reply: Optional[Union[str, Dict, None]] = "", llm_config: Optional[Union[Dict, bool]] = False, system_message: Optional[str] = "")\n')),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("inlineCode",{parentName:"li"},"name")," ",(0,r.kt)("em",{parentName:"li"},"str")," - name of the agent."),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("inlineCode",{parentName:"li"},"is_termination_msg")," ",(0,r.kt)("em",{parentName:"li"},"function"),' - a function that takes a message in the form of a dictionary\nand returns a boolean value indicating if this received message is a termination message.\nThe dict can contain the following keys: "content", "role", "name", "function_call".'),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("inlineCode",{parentName:"li"},"max_consecutive_auto_reply")," ",(0,r.kt)("em",{parentName:"li"},"int"),' - the maximum number of consecutive auto replies.\ndefault to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).\nThe limit only plays a role when human_input_mode is not "ALWAYS".'),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("inlineCode",{parentName:"li"},"human_input_mode")," ",(0,r.kt)("em",{parentName:"li"},"str"),' - whether to ask for human inputs every time a message is received.\nPossible values are "ALWAYS", "TERMINATE", "NEVER".\n(1) When "ALWAYS", the agent prompts for human input every time a message is received.\nUnder this mode, the conversation stops when the human input is "exit",\nor when is_termination_msg is True and there is no human input.\n(2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or\nthe number of auto reply reaches the max_consecutive_auto_reply.\n(3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops\nwhen the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.'),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("inlineCode",{parentName:"li"},"function_map")," ",(0,r.kt)("em",{parentName:"li"},"dict","[str, callable]")," - Mapping function names (passed to openai) to callable functions."),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("inlineCode",{parentName:"li"},"code_execution_config")," ",(0,r.kt)("em",{parentName:"li"},"dict or False")," - config for the code execution.\nTo disable code execution, set to False. Otherwise, set to a dictionary with the following keys:",(0,r.kt)("ul",{parentName:"li"},(0,r.kt)("li",{parentName:"ul"},'work_dir (Optional, str): The working directory for the code execution.\nIf None, a default working directory will be used.\nThe default working directory is the "extensions" directory under\n"path_to_flaml/autogen".'),(0,r.kt)("li",{parentName:"ul"},"use_docker (Optional, list, str or bool): The docker image to use for code execution.\nIf a list or a str of image name(s) is provided, the code will be executed in a docker container\nwith the first image successfully pulled.\nIf None, False or empty, the code will be executed in the current environment.\nDefault is True, which will be converted into a list.\nIf the code is executed in the current environment,\nthe code must be trusted."),(0,r.kt)("li",{parentName:"ul"},"timeout (Optional, int): The maximum execution time in seconds."),(0,r.kt)("li",{parentName:"ul"},"last_n_messages (Experimental, Optional, int): The number of messages to look back for code execution. Default to 1."))),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("inlineCode",{parentName:"li"},"default_auto_reply")," ",(0,r.kt)("em",{parentName:"li"},"str or dict or None")," - the default auto reply message when no code execution or llm based reply is generated."),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("inlineCode",{parentName:"li"},"llm_config")," ",(0,r.kt)("em",{parentName:"li"},"dict or False")," - llm inference configuration.\nPlease refer to ",(0,r.kt)("a",{parentName:"li",href:"/docs/reference/autogen/oai/completion#create"},"autogen.Completion.create"),"\nfor available options.\nDefault to false, which disables llm-based auto reply."),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("inlineCode",{parentName:"li"},"system_message")," ",(0,r.kt)("em",{parentName:"li"},"str")," - system message for ChatCompletion inference.\nOnly used when llm_config is not False. Use it to reprogram the agent.")))}u.isMDXComponent=!0}}]);