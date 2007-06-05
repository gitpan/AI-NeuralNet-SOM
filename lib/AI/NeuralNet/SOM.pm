package AI::NeuralNet::SOM;

use strict;
use warnings;

require Exporter;
use base qw(Exporter);

use Data::Dumper;

=pod

=head1 NAME

AI::NeuralNet::SOM - Perl extension for Kohonen Maps

=head1 SYNOPSIS

  use AI::NeuralNet::SOM;
  my $nn = new AI::NeuralNet::SOM (output_dim => "5x6",
                                     input_dim  => 3);
  $nn->initialize;
  $nn->train (30, 
    [ 3, 2, 4 ], 
    [ -1, -1, -1 ],
    [ 0, 4, -3]);

  print $nn->as_data;

=head1 DESCRIPTION

This package is a stripped down implementation of the Kohonen Maps
(self organizing maps). It is B<NOT> meant as demonstration or for use
together with some visualisation software. And while it is not (yet)
optimized for speed, some consideration has been given that it is not
overly slow.

Particular emphasis has be given that the package plays nicely with
others. So no use of files, no arcane dependencies, etc.

=head2 Scenario

The basic idea is that the neural network consists of a 2-dimensional
array of N-dimensional vectors. When the training is started these
vectors may be complete random, but over time the network learns from
the sample data, also N-dimensional vectors.

Slowly, the vectors in the network will try to approximate the sample
vectors fed in. If in the sample vectors there were clusters, then
these clusters will be neighbourhoods within the rectangle.

Technically, you have reduced your dimension from N to 2.

=head1 INTERFACE

=head2 Constructor

The constructor takes arguments:

=over

=item C<output_dim> : (mandatory, no default)

A string of the form "3x4" defining the X and the Y dimensions.

=item C<input_dim> : (mandatory, no default)

A positive integer specifying the dimension of the sample vectors (and
hence that of the vectors in the grid).

=item C<learning_rate>: (optional, default C<0.1>)

This is a magic number which influence how strongly the vectors in the
grid can be influenced. Higher movement can mean faster learning if
the clusters are very pronounced. If not, then the movement is like
noise and the convergence is not good. To mediate that effect, the
learning rate is reduced over the iterations.

=back

Example:

    my $nn = new AI::NeuralNet::SOM (output_dim => "5x6",
				     input_dim  => 3);

=cut

sub new {
    my $class = shift;
    my %options = @_;
    my $self = bless { %options }, $class;

    if ($self->{output_dim} =~ /(\d+)x(\d+)/) {
	$self->{_X} = $1 and $self->{_Y} = $2;
    } else {
	die "output dimension does not have format MxN";
    }
    if ($self->{input_dim} > 0) {
	$self->{_Z} = $self->{input_dim};
    } else {
	die "input dimension must be positive integer";
    }

    ($self->{_Sigma0}) = map { $_ / 2 } sort {$b <= $a } ($self->{_X}, $self->{_Y});     # impact distance, start value
    $self->{_L0} = $options{learning_rate} || 0.1;                                       # learning rate, start value

    return $self;
}

=pod

=head2 Methods

=over

=item I<initialize>

I<$nn>->initialize

You need to initialize all vectors in the map.

By default, the vectors will be initialized with random values, so all
point chaotically into different directions.  This may not be overly
clever as it may slow down the convergence process unnecessarily.

TODO: provide more flexibility to initialize with eigenvectors

=cut

sub _randomized {
    return rand( 1 ) - 0.5;
}

sub _zero {
    return 0;
}

sub initialize {
    my $self = shift;
    my $meth = shift || \&_randomized;

    for my $x (0 .. $self->{_X}-1) {
	for my $y (0 .. $self->{_Y}-1) {
	    $self->{map}->[$x]->[$y] = [ map { &$meth() } 1..$self->{_Z} ];
	}
    }
}

=pod

=item I<train>

I<$nn>->train ( I<$epochs>, I<@samples> )

The training uses the sample vectors to make the network learn. Each
vector is simply a reference to an array of values.

The C<epoch> parameter controls how often the process is repeated.

Example:

   $nn->train (30, 
               [ 3, 2, 4 ],
               [ -1, -1, -1 ], 
               [ 0, 4, -3]);

TODO: expose distance

=cut


sub _distance { 
    my ($V, $W) = (shift,shift);

#
#                       __________________
#                      / n-1          2
#        Distance  =  /   E  ( V  -  W )
#                   \/    0     i     i
#

#warn "bef dist ".Dumper ($V, $W);
    my $d2 = 0;
    map { $d2 += $_ }
        map { $_ * $_ }
	map { $V->[$_] - $W->[$_] } 
        (0 .. $#$W);
#warn "d2 $d2";
    return sqrt($d2);
}


sub _bmu {                                                                     # http://www.ai-junkie.com/ann/som/som2.html
    my $self   = shift;
    my $sample = shift;

    my $closest;                                                               # [value, x,y] value and co-ords of closest match
    for my $x (0 .. $self->{_X}-1) {
	for my $y (0 .. $self->{_Y}-1){
	    my $distance = _distance ($self->{map}->[$x]->[$y], $sample);
#warn "distance $x $y: $distance";
	    $closest = [$distance,0,0]   unless $closest;
	    $closest = [$distance,$x,$y] if $distance < $closest->[0];
#warn "closest ".Dumper $closest;
	}
    }
    return ($closest->[1], $closest->[2]);
}


sub _neighbors {                                                               # http://www.ai-junkie.com/ann/som/som3.html
    my $self = shift;
    my $sigma = shift;
    my $X     = shift;
    my $Y     = shift;     

    my @neighbors;
    for my $x (0 .. $self->{_X}-1) {
        for my $y (0 .. $self->{_Y}-1){
            my $distance = sqrt ( ($x - $X) ** 2 + ($y - $Y) ** 2 );
	    next if $distance > $sigma;
	    push @neighbors, [ $distance, $x, $y ];                                    # we keep the distances
	}
    }
    return \@neighbors;
}


sub _adjust {                                                                           # http://www.ai-junkie.com/ann/som/som4.html
    my $self  = shift;
    my $l     = shift;                                                                  # the learning rate
    my $sigma = shift;                                                                  # the current radius
    my $unit  = shift;                                                                  # which unit to change
    my ($d, $x, $y) = @$unit;                                                           # it contains the distance
    my $v     = shift;                                                                  # the vector which makes the impact

    my $w     = $self->{map}->[$x]->[$y];                                               # find the data behind the unit
    my $theta = exp ( - ($d ** 2) / (2 * $sigma ** 2));                                 # gaussian impact (using distance and current radius)

    foreach my $i (0 .. $#$w) {                                                         # adjusting values
	$w->[$i] = $w->[$i] + $theta * $l * ( $v->[$i] - $w->[$i] );
    }
}


sub train {
    my $self   = shift;
    my $epochs = shift || 1;

    $self->{LAMBDA} = $epochs / log ($self->{_Sigma0});                                 # educated guess?

    for my $epoch (1..$epochs){
	$self->{T} = $epoch;
	my $sigma = $self->{_Sigma0} * exp ( - $self->{T} / $self->{LAMBDA} );          # compute current radius
	my $l     = $self->{_L0}     * exp ( - $self->{T} / $epochs );                  # current learning rate

	my $sample = @_ [ int (rand (scalar @_) ) ];                                    # take random sample

	my @bmu = _bmu ($self, $sample);                                                # find the best matching unit
#warn "bmu ".Dumper \@bmu;
	my $neighbors = _neighbors ($self, $sigma, @bmu);                               # find its neighbors
#warn "neighbors ".Dumper $neighbors;
	map { _adjust ($self, $l, $sigma, $_, $sample) } @$neighbors;                   # bend them like Beckham
    }
}

=pod

=item I<radius>

I<$radius> = I<$nn>->radius

Returns the initial I<radius> of the map.

=cut

sub radius {
    my $self = shift;
    return $self->{_Sigma0};
}

=pod

=item I<map>

I<$m> = I<$nn>->map

This method returns the 2-dimensional array of vectors in the grid (as
a reference to an array of references to arrays of vectors.

Example:

   my $m = $nn->map;
   for my $x (0 .. 5) {
       for my $y (0 .. 4){
           warn "vector at $x, $y: ". Dumper $m->[$x]->[$y];
       }
   }

=cut

sub map {
    my $self = shift;
    return $self->{map};
}

=pod

=item I<as_string>

print I<$nn>->as_string

This methods creates a pretty-print version of the current vectors.

=cut

sub as_string {
    my $self = shift;
    my $s = '';

    $s .= "    ";
    for my $x (0..$self->{_X}){
	$s .= sprintf ("   %02d ",$x);
    }
    $s .= sprintf "\n","-"x107,"\n";
    
    my $dim = scalar @{ $self->{map}->[0]->[0] };
    
    for my $x (0 .. $self->{_X}-1) {
	for my $w ( 0 .. $dim-1 ){
	    $s .= sprintf ("%02d | ",$x);
	    for my $y (0 .. $self->{_Y}-1){
		$s .= sprintf ("% 2.2f ", $self->{map}->[$x]->[$y]->[$w]);
	    }
	    $s .= sprintf "\n";
	}
	$s .= sprintf "\n";
    }
    return $s;
}

=pod

=item I<as_data>

print I<$nn>->as_data

This methods creates a string containing the raw vector data, row by
row. This can be fed into gnuplot, for instance.

=cut

sub as_data {
    my $self = shift;
    my $s = '';

    my $dim = scalar @{ $self->{map}->[0]->[0] };
    for my $x (0 .. $self->{_X}-1) {
	for my $y (0 .. $self->{_Y}-1){
	    for my $w ( 0 .. $dim-1 ){
		$s .= sprintf ("\t%f", $self->{map}->[$x]->[$y]->[$w]);
	    }
	    $s .= sprintf "\n";
	}
    }
    return $s;
}

=pod

=back


=head1 SEE ALSO

L<http://www.ai-junkie.com/ann/som/som1.html>

=head1 AUTHOR

Robert Barta, E<lt>rho@devc.atE<gt>

=head1 COPYRIGHT AND LICENSE

Copyright (C) 2007 by Robert Barta

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.8.8 or,
at your option, any later version of Perl 5 you may have available.

=cut

our $VERSION = '0.01';

1;

__END__


