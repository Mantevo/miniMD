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
    echo "usage: $(basename $0) nthreads"
    exit 1
fi

nthreads=$1
for var in nooffload nvptx # x86
do
    build_count=0
    build_success=0
    run_count=0
    run_success=0
    errfile=$(mktemp -t "$(basename $0).XXXX")
    outfile=$(mktemp -t "$(basename $0).XXXX")
    printf "Trying OFFLOAD=$var..."
    build_count=$((build_count + 1))
    make TOOLS=clang PAD=4 SP=yes OFFLOAD="$var" -j 36 -B 2> ${errfile} > ${outfile}
    if [ $? -eq 0 ]
    then
        printf "${GREEN}BUILD PASSED${NORMAL}!\n"
        build_success=$((build_success + 1))
        for halfneigh in 0 1
        do
            resfile=$(mktemp -t "$(basename $0).XXXX")
            printf "Running taskset -c 0-${nthreads} ./miniMD_clang -t ${nthreads} --half_neigh ${halfneigh} "
            run_count=$((run_count + 1))
            taskset -c 0-${nthreads} ./miniMD_clang -t ${nthreads} --half_neigh ${halfneigh} --check-output ${resfile} 2> ${errfile} > ${outfile}
            if [ $? -ne 0 ]
            then
                printf "  ${RED}RUN FAILED${NORMAL}!\n"
            else
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
            fi
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
