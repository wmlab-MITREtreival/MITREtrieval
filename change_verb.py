def change_verb(verb):
    # install, survey,'detect','detects','download', 'navigate','locate','issue','move','take','click','alerts','entrench','exfiltrate', 'deploy', 'putfile','compose','create', 'creates','copy','copies','save','saved','saves','allocate','assign'
    v_unlink = ['delete', 'clear', 'remove', 'erase', 'wipe','purge','expunge', 'drop','drops']
    v_write = ['store', 'place', 'write','add','adds','modify','modifies','append','appends','record','records','set']
    v_read = ['read','obtain','acquire','check','checks']
    v_exec = [ 'use', 'execute', 'executed', 'run', 'ran', 'launch', 'call', 'perform', 'list', 'invoke', 'inject', 'implant', 'open', 'opened','target','resume','exec','e ute','ute','e uted']
    v_fork = ['clone', 'clones','spawned','spawn','spawns', 'fork']
    # v_setuid = ['elevate', 'impersonated']
    v_send = ['send', 'sent','transfer','post','postsinformation','postsinformations', 'transmit','deliver','push','redirect','redirects']
    v_receive = ['receive','accept','get','gets']
    v_collect = ['collect', 'gather', 'extract','extracts']
    v_connect = ['browse', 'browses', 'connect', 'connected', 'portscan', 'connects','communicates','communicate']
    v_chmod = ['chmod', 'change permission','changes permission', 'permision-modifies', 'modifies permission','modify permission']
    v_load = ['load', 'loads']
    v_exit = ['terminate', 'terminates','stop','stops','end','finish','break off','abort','conclude']
    
    if verb in v_unlink:
        return "remove"
    elif verb in v_write:
        return "write"
    elif verb in v_read:
        return "read"
    elif verb in v_exec:
        return "execute"
    elif verb in v_fork:
        return "fork"
    elif verb in v_send:
        return "send"
    elif verb in v_receive:
        return "receive"
    elif verb in v_connect:
        return "connect"
    elif verb in v_chmod:
        return "chmod"
    elif verb in v_load:
        return "load"
    elif verb in v_exit:
        return "exit"
    elif verb in v_collect:
        return "collect"
    elif verb=='instal':
        return "install"
    else:
        return verb

if __name__=='__main__':
    print(change_verb('gather'))