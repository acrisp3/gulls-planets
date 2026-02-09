#!/usr/bin/perl -w

use strict;
use POSIX;


my $name="test_grid_draw"; # run name
my $npf=100; # num planets per file
my $nr=1; # num files per field, i.e., subruns

my $mearth=3.00374072e-6;
my $pi=4.0*atan(1);

if(! -d "$name/")
{
    mkdir("$name");
}
chdir("$name/");

open(FLD,"<$ENV{'GULLS_STARS_DIR'}/Huston2023_surot2d/gulls_surot2d_H2023.sources") || die "Could not open sources file\n";

#perl function to read succesive lines from file
while(my $line=<FLD>)
{
    #remvoes eol
    chomp($line);
    #split lines by spaces into array
    my @data = split(' ',$line);
    #if length of aray is greater than7, things didnt split right or not right number of columns
    if(@data!=7){next;}
        #make f equal to first element of array (field number)
    my $f = $data[0];

    #gen filename base
    my $base = "$dir/${rundes[$r]}.planets.$f.";

    print "Writing $base files with $nl lines\n";


#for r less than length of rundes
for (my $r=0; $r<@rundes; $r++)
{
    my $dir = "${rundes[$r]}";
    #make dir with rundes as name in directory you run from
    if(! -d $dir)
    {
        mkdir($dir);
    }
    #XXX probably need to check
    open(FLD,"<$ENV{'GULLS_BASE_DIR'}/starfields/Huston2023_surot2d/gulls_surot2d_H2023.sources") || die "Could not open sources file\n";
    #open(FLD,"<$ENV{'GULLS_BASE_DIR'}/kgriz/KgrizVRIw.sources") || die "Could not open sources file\n";
    #open(FLD,"</home/penny/dmabuls/kgriz/KgrizVRIw.sources") || die "Could not open sources file\n";

    #rundes.sightline.subrun or some combo
    #looping line over list of sightlines or fields
    
    #perl function to read succesive lines from file
    while(my $line=<FLD>)
    {
	#remvoes eol
	chomp($line);
	#split lines by spaces into array
	my @data = split(' ',$line);
	#if length of aray is greater than7, things didnt split right or not right number of columns
	if(@data!=7){next;}
	#make f equal to first element of array (field number)
	my $f = $data[0];
	
	#gen filename base
	my $base = "$dir/${rundes[$r]}.planets.$f.";
	
	print "Writing $base files with $nl lines\n";

	#loop over number of subruns
	for(my $i=0;$i<$nf;$i++)
	{
	    
	    $pfile = "$base$i";
	    #if file exists, don't bother generating
	    if( -e $pfile){next;}
	    #other open up and write to file
	    open(OUT,">$pfile") || die "Could not open output file $pfile\n";
	    
	    #now loop over the number of events generated
	    for(my $j=0;$j<$nl;$j++)
	    {
		$a = 10**($amin + ($amax-$amin)*rand());
		$mass = 3.00374072e-6*10**($mmin + rand()*($mmax-$mmin));
		#$a = 1+int(3*rand())/2;
		my $rnd = rand();
		$inc = 180*($rnd<0.5?acos(2*$rnd):-acos(2-2*$rnd))/$pi;
		#$inc = -90.0 + 180.0*rand();
		$p = 360.0*rand();
		print OUT "$mass $a $inc $p\n";
	    } #end for nlines

	    close(OUT);
	} #end for nfiles
    } #end for fields
    close(FLD);
} #end for rundes

