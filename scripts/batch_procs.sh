# Shared process matcher for run_batch.sh and stop_batch.sh.  Source, do not run.
#
# Matching on the command line alone is not safe here.  Any shell, grep, or editor
# whose argv happens to contain "main.py --links" matches too, and signalling
# *that* process group kills the wrong thing -- which is exactly what a first cut
# of stop_batch.sh did to its own caller.  Requiring the command name to be a
# Python interpreter, and excluding this shell's own process tree, restricts the
# match to processes that are actually running the batch.
#
# The input flag is matched as an alternation rather than a fixed string: the
# batch used to be started with --s3-prefix and is now started with --links, and
# a matcher that knew only the old one would report "nothing running" while 5
# workers held the GPU -- so stop_batch.sh would exit clean and leave them.

_batch_rows() {
    ps -eo pid=,ppid=,pgid=,comm=,args= \
        | awk -v self="$$" -v ppid="$PPID" '
            $4 ~ /^python/ \
            && /main\.py/ \
            && /--(links|manifest|s3-prefix)/ \
            && $1 != self && $1 != ppid
        '
}

batch_pids()   { _batch_rows | awk '{print $1}'; }
batch_count()  { _batch_rows | wc -l | tr -d ' '; }
batch_pgid()   { _batch_rows | awk '{print $3; exit}'; }
batch_orphans(){ _batch_rows | awk '$2 == 1 {print $1}'; }
batch_show()   { _batch_rows | cut -c1-120; }
