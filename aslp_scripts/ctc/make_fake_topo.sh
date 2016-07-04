#!/bin/bash

# Created on 2016-03-04
# Author: Binbin Zhang
self_jump=0.5

echo "$0 $@"  # Print the command line for logging
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
    echo "Make mono state fake topo"
    echo "Usage: $0 num_state out_topo_file"
    echo "eg: $0 215 fake.topo"
    exit -1;
fi

num_state=$1
topo_file=$2

other_jump=$(echo "1-$self_jump" | bc)

{
echo "<Topology>"
echo "<TopologyEntry>"
echo "<ForPhones>"
for((i=1; i<=$num_state; i++)); do 
    echo -n "$i "
done
echo ""
echo "</ForPhones>"
echo "<State> 0 <PdfClass> 0 <Transition> 0 $self_jump <Transition> 1 $other_jump </State>"
echo "<State> 1 </State>"
echo "</TopologyEntry>"
echo "</Topology>"
} > $topo_file


