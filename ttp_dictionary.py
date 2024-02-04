from query_list import *
import json

def get_authentication():
    with open('Data/neo4j_info.json') as f:
        neo4j_info = json.load(f)
    f.close()
    return neo4j_info

must_decide = ['t1014', 't1115', 't1091', 't1486', 't1197', 't1221', 't1220', 't1490', 't1505', 't1548', 't1561','t1008','t1010']
may_decide = must_decide + ['t1140', 't1027', 't1113', 't1218', 't1056', 't1059', 't1047', 't1055', 't1547','t1189','t1559','t1046']
decide_list = may_decide # must_decide/may_decide

# all tech name mapping
must_ttp_dict = {
    # initial access
    't1189' : ['watering hole'],
    't1091' : [' usb ', 'removable media', 'removable drive','removable data storage','removable disk'],
    't1566' : ['malicious email'],

    # TA0002 Execution
    't1047' : ['windows management instrumentation', 'wmi'],
    't1059' : ['powershell', 'applescript', 'cmd', 'vbs', 'visual basic script', 'visualbasic', 'python', 'javascript', 'shellcode','jscript','vba',' batch scripts'],
    't1559' : [' ipc ', 'inter-process communication','inter process communication',' dde ','Dynamic Data Exchange','Object Linking and Embedding ',' OLE ','DCOM'],
    't1106' : ['ntcreateprocess', 'windows api', 'winapi'],
    't1569' : ['launchctl', 'services.exe'],
    't1204' : ['link attached to the email','the victim click','opening the link','open the link','lure document','malicious link'],
    't1129' : ['LoadLibrary'],
    # TA0003 Persistence
    't1197' : ['bits protocol', 'bitsadmin', 'bits utility', 'bits job'],
    't1547' : ['startup folder', 'startup', 'registry run key', 'authentication package', 'autostart', 'active setup', 'shortcut','registry key'],
    't1543' : ['launchagent', 'launch agent', 'launchdaemon', 'launch daemon', 'new service','systemd service'],
    't1574' : ['dll side loading' , 'dll hijack', 'ld_preload', 'load dll'],
    't1542' : ['bootkit', 'TFTP', 'trivial file transfer protocol'],
    't1053' : ['scheduled task', 'task scheduler','systemd timer','tasks scheduler'],
    't1505' : ['sql stored procedure', 'transport agent','web shell', 'iis'],
    't1078' : ['legitimate credential', 'valid credential', 'valid account', 'valid ssh', 'valid vpn'],

    # TA0004 Privilege Escalation
    't1548' : ['uac', 'user account control', 'authorizationexecutewithprivileges'],
    't1134' : ['sid-history' , 'sid history', 'runas', 'createprocesswithtoken', 'logonuser', 'setthreadtoken', 'sedebugprivilege', 'adjusttokenprivilege','token manipulation','Impersonation','Impersonate','PID spoofing'],
    't1484' : ['gpo', 'group policy object'],
    't1055' : ['inject','hollow'],
    't1546' : ['AppCert DLL','AppInit DLL','screensaver'],

    # TA0005 Defense Evasion
    't1014' : ['rootkits', 'rootkit'],
    't1221' : ['template injection', 'Template'],
    't1140' : ['deobfuscate', 'decrypt', 'decode'], # Deobfuscate/Decode Files or Information, ['AES', 'XOR', ' RC4']
    't1480' : ['environmental key'],
    't1564' : ['mktemp', 'hidden file', 'hidden folder', 'hidden directory', 'hidden directories', 'hidden file system','hide file','hide folder','NTFS','conceal PowerShell window','hidden windows'],
    't1036' : ['Masquerad', 'disguise', 'right-to-left-override', 'right-to-left override', 'rtlo'], # little危
    't1027' : ['packed via', 'Obfuscate', 'encrypt', 'encode', 'Binary Padding'], # Obfuscated Files or Information ['XOR', 'RC4','base64', 'AES']
    't1562' : ['disable winnows defender','disabling Windows Defender','disabled Windows Defender','stop anti-malware','stopped anti-malware','terminate antimalware process','disable security product','disabled security product','disable anti-virus','disabled event log','disable event log'],
    't1070' : ['timestomp','./httpd-nscache_clean'],
    't1112' : ['modify Registry', 'modifying Registry','modifies Registry'],
    't1620' : ['reflectively load', 'Reflective Code'],
    't1218' : ['Rundll32', 'MMC', 'Microsoft Management Console', 'Regsvr32', 'odbcconf', 'Msiexec', 'Mshta', 'InstallUtil', 'CMSTP', 'Mavinject', 'Verclsid', 'Regsvcs', 'Regasm','rundll32'],
    't1127' : ['MSBuild'],
    't1550' : ['Pass the Hash', 'Pass the Ticket','delete cookie','remove cookie','deletes cookie','removes cookie'],
    't1497' : ['anti-sandbox','anti-virtual','evade sandbox','avoid sandbox','anti-debug'],
    't1220' : ['XSL'],
    't1222' : ['chmod','modify permission','modifies permission','icacls','ACL permission'],
    #'t1553' : ['establish the trust'],

    # TA0006 Credential Access
    't1003' : ['lsass', 'Credential Dumping', 'LSA Secrets', 'NTDS', 'Security Account Manager', 'dump credential','dumping credential'],
    't1056' : ['keystrokes', 'keylog', 'hook'],
    't1110' : ['Credential Stuffing', 'brute-force', 'brute force', 'Password Spraying', 'Password Cracking', 'Password Guessing','brute - force'],
    't1555' : ['Keychain', 'Windows Credential Manager', 'password manager', 'recover password','recovered password','recovers password'],
    't1539' : ['Web Session Cookie', 'cookie'],
    't1111' : ['Two-Factor Authentication', '2FA', 'multi-factor authentication', 'MFA'],

    # TA0007	Discovery
    't1016' : ['identify the MAC address','gather network configuration','list of hosts or ip addresses','list of ip addresses','ip address of the victim','netmon.exe','ipconfig', 'ifconfig','collect MAC address','collect the network adapter information','GetAdaptersInfo','enumerate system network','enumerate the IP address','collect DNS information','list of the system network interface','network setting'],
    't1057' : ['list of processes','list running processes','list of running processes', 'show running process','running process','tasklist','names of running process','list process', 'ProcessList'],
    't1083' : ['list of files','list file', 'browse file systems'],
    't1518' : ['antivirus software','antivirus process'],
    't1082' : ['os version','system information','operating system version','version of the operating system','version of  operating system'],
    't1012' : ['enumerate registry', 'list registry','queries'],
    't1124' : ['getlocaltime','time zone','system time','collect the timestamp','collect the time','check the system time','get_current_time','GetSystemTime'],
    't1046' : ['network scan','port scan','scanned network service','scan for open port','scans for open port','Network Scanner','scan IP','GetHttpsInfo','netbios scanner','server scan','nbtscan'],
    't1135' : ['network share','net share','query shared drive','NetShareEnum'],
    't1518' : ['installed software', 'enumerate software', 'list software'],
    't1082' : ['OS version', 'system information'],
    't1614' : ['geographical location', 'geographical information'],
    't1049' : ['netstat','net use','net session','netmon.exe'],
    't1033' : ['whoami', 'username', 'user name', 'hostname', 'username','computername','user list','PC name','quser'],
    't1124' : ['timestamp', 'local time', 'system time', 'time zone', 'current time'],
    't1069' : ['domain group','net group','net localgroup','local group'], #QQ
    't1018' : ['net view','detect remote system'],
    't1482' : ['Domain trusts','Get-NetDomainTrust'],
    't1010' : ['the titles of running window','open windows','Duqu','foreground window','enumerate windowa','window name','windows name','EnumWindows()','windows title','window title','Window enumeration','active window'],
    # TA0008 Lateral Movement : 0
    't1021' : ['Remote Desktop Protocol', 'RDP','DCOM'],
    't1570' : ['moving laterally','move laterally'],
    't1210' : ['ZeroLogon','EternalBlue','EternalRomance','BlueKeep'],

    # TA0009 Collection
    't1115' : ['clipboard'], 
    't1123' : ['audio captur', 'recording audio', 'record audio', 'capturing audio', 'capture audio', 'captured audio', 'captures audio', 'performing audio', 'perform audio','audio record'],
    't1113' : ['screenshot', 'screen captur', 'screen shot','capture screen','capturing screen','captures screen','victim screen'],
    't1039' : ['from network share','from file share','network shared drive','shared drives'], #QQ
    # TA0010 Exfiltration
    't1041' : ['exfiltrates data over the C2 channel', 'from c2 server', 'to the c2 server','over the c2 channel','to a c2 server', 'over the C2 channel'],
    't1048' : ['dns tunneling for exfiltration', 'udp for exfiltration,exfiltrates files over ftp', 'exfiltrates files over http', 'exfiltrates files over udp', 'ftp to exfiltrate'],
    't1567' : ['exfiltrate over google Drive'],

    # TA0011 Command and Control
    't1071' : ['http communication', 'rdp connection', 'http for c2 communication', 'ftp protocol for c2 communication', 'ftp protocol' , 'smtp for c2 communication', 'dns for c2 communication','http request','get request','post request','http traffic','POST HTTP request',' HTTPS protocol','HTTP POST request','HTTP get request'],
    't1568' : ['dynamic dns','fast flux', 'dga', 'domain generation algorithm', 'dns calculation','DNS resolution'],
    't1573' : ['encryption for c2 communication', 'encrypt channel','encrypted channel', 'encryption of command and control traffic', 'encrypt c2 traffic','encrypted c2 traffic', 'encrypts c2 communication', 'symmetric encryption', 'asymmetric encryption'],
    't1095' : ['icmp', 'tcp for c2', 'udp for c2', 'tcp protocol for c2','custom TCP protocol','SSL to communicate'],
    't1572' : ['SSH tunnel','FRPC.exe'],
    't1102' : ['one-way communication','dead drop resolver',' ddr '],
    't1008' : ['fallback'],
    't1105' : ['from a remote server'],
    't1219' : ['TeamViewer','AmmyyAdmin','VNC module','vncDll module','LogMein'],
    # TA0040 Impect
    't1490' : ['shadow'],
    't1486' : ['ransom'],
    't1561' : ['Master Boot Record', 'MBR']
}

# greeter = Ontology("bolt://140.115.54.90:10096", "neo4j", "wmlab")
neo4j_info = get_authentication()
greeter = Ontology(neo4j_info["url"],neo4j_info["account"], neo4j_info["password"])
tech_id_name_dict = greeter.get_all_tech_id_name()
greeter.close()

def rule_based_ttp(document):
    global must_ttp_dict
    global all_tech_name
    global decide_list
    has_ttp = []
    rm_ttp = []
    
    for key, values in must_ttp_dict.items():
        for v in values:
            if v.lower() in document:
                has_ttp.append(key)
                break

    for key, value in tech_id_name_dict.items():
        if value in document:
            has_ttp.append(key)

    has_ttp = list(set(has_ttp))
    total_ttp = list(set(list(must_ttp_dict.keys())))

    for tid in decide_list:
        if not(tid in has_ttp):
            rm_ttp.append(tid)
    
    return total_ttp , has_ttp, rm_ttp

if __name__=='__main__':
    doc = "tood screen shot domain group rdp Disk Wipe."
    total_ttp , has_ttp,rm_ttp = rule_based_ttp(doc)
    print(has_ttp)