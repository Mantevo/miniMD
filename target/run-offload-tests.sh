#!/bin/bash

RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
NORMAL=$(tput sgr0)

tmpref="8.200912e-01"
engref="-5.852703e+00"
pref="-1.873937e-01"
tol="1e-1"

if [[ $# -lt 1 ]]
then
    echo "usage: $(basename $0) nthreads [perffile]"
    exit 1
fi

nthreads=$1
sizes="32"

if [[ $# -eq 2 ]]
then
    perffile=$2
    sizes="32 40 80"
fi

build_count=0
build_success=0
run_count=0
run_success=0

for var in nooffload nvptx x86
do
    errfile=$(mktemp -t "$(basename $0).XXXX")
    outfile=$(mktemp -t "$(basename $0).XXXX")
    printf "Trying OFFLOAD=$var..."
    build_count=$((build_count + 1))
    make TOOLS=clang PAD=4 SP=yes OFFLOAD="$var" -j 36 -B 2> ${errfile} > ${outfile}
    if [ $? -eq 0 ]
    then
        printf "${GREEN}BUILD PASSED${NORMAL}!\n"
        build_success=$((build_success + 1))

        if [ "${var}" = x86 ]
        then
            maxthreads=`./miniMD_clang --print-user-max-team-size`
            if [ $nthreads -gt $maxthreads ]
            then
                echo "Clamping # of threads for ${var} target (max team size is ${maxthreads}, ${nthreads} requested.)"
                nthreads=$maxthreads
            fi
        fi
        for halfneigh in 0 1
        do
            for ghost_neighbor in 0 1
            do
                for size in ${sizes}
                do
                    infile=./in.lj.miniMD.${size}
                    resfile=$(mktemp -t "$(basename $0).XXXX")
                    if [[ $# -eq 1 ]]
                    then
                        perffile=${errfile}
                    fi
                    printf "Running taskset -c 0-$((nthreads-1)) ./miniMD_clang -t ${nthreads} --half_neigh ${halfneigh} --ghost_neighbor ${ghost_neighbor} -i ${infile}"
                    run_count=$((run_count + 1))
                    taskset -c 0-$((nthreads-1)) ./miniMD_clang -t ${nthreads} --half_neigh ${halfneigh} --ghost_neighbor ${ghost_neighbor} -i ${infile} --check-output ${resfile} 2>> ${perffile} > ${outfile}
                    if [ $? -ne 0 ]
                    then
                        printf "  ${RED}RUN FAILED${NORMAL}!\n"
                    else
                        if [ "${size}" -eq 32 ]
                        then
                            temp=`cut -d' ' -f2 < ${resfile}`
                            eng=`cut -d' ' -f3 < ${resfile}`
                            p=`cut -d' ' -f4 < ${resfile}`

                            tmpchk=`perl -e "print abs((${tmpref} - ${temp})/${tmpref}) < ${tol}"`
                            engchk=`perl -e "print abs((${engref} - ${eng})/${engref}) < ${tol}"`
                            pchk=`perl -e   "print abs((${pref}   - ${p})/${pref}) < ${tol}"`
                            printf " TUP  ${temp} ${eng} ${p} "
                            if [ ! -z ${tmpchk} ]  && [ ! -z ${engchk} ] && [ ! -z ${pchk} ]
                            then
                                printf "  ${GREEN}RUN + CHECK PASSED${NORMAL}!\n"
                                run_success=$((run_success + 1))
                            else
                                printf "  ${RED}CHECK FAILED${NORMAL}!"
                                tmprel=`perl -e "printf \"%e\", abs((${tmpref} - ${temp})/${tmpref})"`
                                engrel=`perl -e "printf \"%e\", abs((${engref} - ${eng})/${engref})"`
                                prel=`perl -e   "printf \"%e\", abs((${pref}   - ${p})/${pref})"`
                                printf " ${tmprel} ${engrel} ${prel}\n"
                            fi
                        else
                            printf "  ${GREEN}RUN OK, CHECK SKIPPED${NORMAL}!\n"
                            run_success=$((run_success + 1))
                        fi
                    fi
                done
            done
        done
    else
        printf "${RED}FAILED!${NORMAL}\n"
    fi
done

echo "${build_success}/${build_count} builds succeeded"
echo "${run_success}/${run_count} run succeeded"
if [ "${build_success}" = "${build_count}" ] && [ "${run_success}" = "${run_count}" ]
then
    exit 0
else
    exit 1
fi
