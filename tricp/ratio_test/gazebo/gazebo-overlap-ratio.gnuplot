reset

# set term pdf font "Times-New-Roman,12" enhanced
# set output 'test.pdf'

set term tikz createstyle size 2.5inch, 2.5inch
set output "gazebo-overlap-ratio.tex"
set key left top

set xtics nomirror out
set ytics nomirror out

# set xlabel "Trimming Ratio"
# set xrange [0:1]
# set xtics 0.1

# set ylabel "Success Rate"
set yrange [0:0.7]
# set ytics 0.2

unset border

# Line style for axes
#set style line 80 lt rgb "#808080"
set style line 80 lt 1 lc rgb "#808080"
# Line style for grid
set style line 81 lt 0  # dashed
set style line 81 lt rgb "#808080"  # grey
set grid back linestyle 81
# set border 0 back linestyle 80
set border 0 back

set key on
set key box
set key Left


# special setting
unset key

#key line style
set style line 1 lc rgb "blue" lt 1 dt 4 lw 3.0
set style line 2 lc rgb "orange" lt 1  dt 1 lw 3.0
set style line 3 lc 23  lt 1

# filled
set style fill solid noborder

# plot the pose
# plot "posetranslation.bat" using 2:xtic(1) notitle w lp lc rgb "red" lw 1.4 pt 12 ps 0.8 dt 4,\
# "posetranslation.bat" using 3 notitle w lp lc rgb "red" lw 1.4 pt 21 ps 0.8 dt 4,\
# "posetranslation.bat" using 4 notitle w lp lc rgb "red" lw 1.4 pt 19 ps 0.8 dt 4,\
# "poserotation.bat" using 2:xtic(1) notitle w lp lc rgb "blue" lw 1.4 pt 12 ps 0.8 dt 1,\
# "poserotation.bat" using 3 notitle w lp lc rgb "blue" lw 1.4 pt 21 ps 0.8 dt 1,\
# "poserotation.bat" using 4 notitle w lp lc rgb "blue" lw 1.4 pt 19 ps 0.8 dt 1,\
# NaN ls 1 title "Translation",\
# NaN ls 2 title "Rotation",\
# NaN with points pt 12 ps 0.8 lc rgb "black" title "Easy Pose",\
# NaN with points pt 21 ps 0.8 lc rgb "black" title "Medium Pose",\
# NaN with points pt 19 ps 0.8 lc rgb "black" title "Hard Pose"

plot "overlaptranslation.bat" using 2:xtic(1) notitle w lp ls 1 pt 1 ps 2.5,\
"overlaptranslation.bat" using 3 notitle w lp ls 1 pt 9 ps 2.5,\
"overlaptranslation.bat" using 4 notitle w lp ls 1 pt 4 ps 2.5,\
"overlaprotation.bat" using 2:xtic(1) notitle w lp ls 2 pt 1 ps 2.5,\
"overlaprotation.bat" using 3 notitle w lp ls 2 pt 9 ps 2.5,\
"overlaprotation.bat" using 4 notitle w lp ls 2 pt 4 ps 2.5
# NaN with points pt 12 ps 2.0 lc rgb "black" title "High",\
# NaN with points pt 21 ps 2.0 lc rgb "black" title "Medium",\
# NaN with points pt 11 ps 2.0 lc rgb "black" title "Low"

