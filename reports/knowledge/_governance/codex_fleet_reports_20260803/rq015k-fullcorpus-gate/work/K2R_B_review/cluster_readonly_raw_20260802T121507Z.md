# Cluster Read-Only Raw Outputs

Timestamp command run locally before the live cluster queries:

```text
$ date -u +%Y-%m-%dT%H:%M:%SZ
2026-08-02T12:15:07Z
```

## sacctmgr association

Command:

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc "sacctmgr show assoc user=u25310231 format=User,Account,Partition,QOS,GrpTRES,GrpCPUs,MaxTRES,MaxJobs,MaxSubmitJobs -P"
```

Raw output:

```text
hostfile_replace_entries: mkstemp: Operation not permitted
update_known_hosts: hostfile_replace_entries failed for <local-home>/.ssh/known_hosts: Operation not permitted
User|Account|Partition|QOS|GrpTRES|GrpCPUs|MaxTRES|MaxJobs|MaxSubmit
u25310231|p_p25310231||cpu-4000_core-l40-16_card-a800-16_card|||||
u25310231|u25310231||normal|||||
u25310231|d_000723||normal|||||
```

## squeue

Command:

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc "squeue -h -o '%i|%j|%u|%t|%P|%M|%D|%C|%m|%R'"
```

Raw output:

```text
hostfile_replace_entries: mkstemp: Operation not permitted
update_known_hosts: hostfile_replace_entries failed for <local-home>/.ssh/known_hosts: Operation not permitted
2068003|v4_headw_r1|u25310231|R|L40|4:26:05|1|1|64G|gpu4027
1942893|empc_2x44_gate3|u25310231|PD|intel|0:00|1|1|4G|(DependencyNeverSatisfied)
1942895|empc_2x44_xfer3|u25310231|PD|intel|0:00|1|2|8G|(Dependency)
```

## sinfo aggregate

Command:

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc "sinfo -h -o '%P|%D|%T|%c|%m|%e|%C|%N'"
```

Raw output:

```text
hostfile_replace_entries: mkstemp: Operation not permitted
update_known_hosts: hostfile_replace_entries failed for <local-home>/.ssh/known_hosts: Operation not permitted
intel*|1|drained*|96|644000|N/A|0/0/96/96|cpui190
intel*|183|down*|96|644000|455885-N/A|0/0/17568/17568|cpui[003-119,123,128,133-150,209-222,225-256]
intel*|41|mixed|96|644000|450280-596606|1419/2517/0/3936|cpui[120-121,125,129-132,152-159,161-176,181-187,189,191-192]
intel*|7|allocated|96|644000|443417-546598|672/0/0/672|cpui[122,124,126-127,151,160,188]
amd|1|drained*|192|772000|504661|0/0/192/192|cpua094
amd|158|down*|192|772000|567985-N/A|0/0/30336/30336|cpua[061-070,072-086,098,109-141,149-150,152,154-180,212-220,225-280,293-296]
amd|62|mixed|192|772000|105401-713634|2800/9104/0/11904|cpua[001-007,009-011,071,087-090,092-093,096-097,099-108,142-146,151,153,183-191,193-206,208,210-211]
amd|7|allocated|192|772000|346758-701487|1344/0/0/1344|cpua[008,012,181-182,192,207,209]
amd|52|idle|192|772000|682931-724856|0/9984/0/9984|cpua[013-060,091,095,147-148]
fata|2|inval|192|3094000|122338-123181|0/0/384/384|fata[14,17]
fata|13|drained*|192|3094000|3006944-N/A|0/0/2496/2496|fata[03,05-06,09-13,15-16,18-20]
fata|2|mixed|192|3094000|1809909-2834391|255/129/0/384|fata[01-02]
fata|1|idle|192|3094000|2947573|0/192/0/192|fata04
L40|1|draining|56|1031000|743364|35/0/21/56|gpu4025
L40|21|mixed|56|1031000|186518-949054|674/502/0/1176|gpu[4001,4004-4008,4010-4013,4015-4016,4019-4020,4022,4027-4029,4034-4036]
L40|11|allocated|56|1031000|536058-969515|616/0/0/616|gpu[4003,4009,4014,4017-4018,4023-4024,4026,4037,4041-4042]
L40|1|idle|56|1031000|938679|0/56/0/56|gpu4002
L40-fbb|2|allocated|56|1031000|943544-971710|112/0/0/112|gpu[4038-4039]
A800|4|draining|56|1031000|420202-927895|224/0/0/224|gpu[8008,8017-8019]
A800|1|drained|56|1031000|896804|0/0/56/56|gpu8010
A800|11|mixed|56|1031000|436705-943213|338/278/0/616|gpu[8001-8002,8004-8005,8007,8009,8011-8013,8015,8020]
A800|3|allocated|56|1031000|576498-908136|168/0/0/168|gpu[8003,8006,8014]
```

## fata node detail used for placement sanity check

Command:

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc "sinfo -N -h -p intel,fata,amd -o '%P|%N|%T|%c|%m|%e|%C'"
```

Raw output subset used for fata placement:

```text
fata|fata01|mixed|192|3094000|1809909|191/1/0/192
fata|fata02|mixed|192|3094000|2834391|64/128/0/192
fata|fata04|idle|192|3094000|2947573|0/192/0/192
```

## K1b sacct

Command:

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc "sacct -j 2068976 --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,NodeList -P"
```

Raw output:

```text
hostfile_replace_entries: mkstemp: Operation not permitted
update_known_hosts: hostfile_replace_entries failed for <local-home>/.ssh/known_hosts: Operation not permitted
JobID|JobName|State|ExitCode|Elapsed|AllocCPUS|ReqMem|MaxRSS|NodeList
2068976|zxc-rq015k-k1b|COMPLETED|0:0|00:14:09|16|160G||fata02
2068976.batch|batch|COMPLETED|0:0|00:14:09|16|||fata02
2068976.extern|extern|COMPLETED|0:0|00:14:09|16|||fata02
```
