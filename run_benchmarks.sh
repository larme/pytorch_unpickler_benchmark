mkdir -p outputs && rm -f outputs/*.csv && \
    python benchmark.py --disable-gpu --function-name=loads && \
    python benchmark.py --disable-gpu --function-name=_fix_torch_loads && \
    python benchmark.py --disable-gpu --function-name=loads_or_fix_torch && \
    python benchmark.py --function-name=loads && \
    python benchmark.py --function-name=_fix_torch_loads && \
    python benchmark.py --function-name=loads_or_fix_torch && \
    echo "function,data,repeat,total time,avg time,cuda availablity" > all.csv && \
    cat outputs/*.csv >> all.csv

