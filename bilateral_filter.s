	.file	"bilateral_filter.cpp"
	.text
.Ltext0:
	.section	.text.unlikely._Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.0,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LCOLDB0:
	.section	.text._Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.0,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LHOTB0:
	.p2align 4,,15
	.section	.text.unlikely._Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.0,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.Ltext_cold0:
	.section	.text._Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.0,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
	.type	_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.0, @function
_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.0:
.LFB12399:
	.file 1 "bilateral_filter.cpp"
	.loc 1 120 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
.LVL0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rdi, %rbx
	subq	$72, %rsp
	.cfi_def_cfa_offset 128
.LBB5697:
	.loc 1 125 0
	movl	80(%rdi), %r14d
.LBE5697:
	.loc 1 120 0
	movq	40(%rdi), %r12
.LVL1:
	movq	%fs:40, %rax
	movq	%rax, 56(%rsp)
	xorl	%eax, %eax
	movq	32(%rdi), %rbp
.LVL2:
.LBB5828:
	.loc 1 125 0
	leal	3(%r14), %ecx
	testl	%r14d, %r14d
	cmovns	%r14d, %ecx
	sarl	$2, %ecx
	movl	%ecx, 8(%rsp)
	call	omp_get_num_threads
.LVL3:
	movl	%eax, %r15d
	movl	%eax, 20(%rsp)
	call	omp_get_thread_num
.LVL4:
	movl	8(%rsp), %ecx
	movl	%eax, %r10d
	movl	%eax, 16(%rsp)
	imull	%ecx, %r10d
	leal	(%rcx,%r10), %r13d
	cmpl	%r14d, %r13d
	cmovg	%r14d, %r13d
	cmpl	%r10d, %r14d
	jle	.L2
	movl	%r15d, %edi
	movl	%eax, %r11d
	movl	96(%rbx), %edx
	imull	%ecx, %edi
	movl	84(%rbx), %eax
	movl	%edi, 24(%rsp)
	movl	%r15d, %edi
	addl	%r11d, %edi
	movl	%edi, %esi
	movl	%edi, %r15d
	addl	$1, %esi
	imull	%ecx, %r15d
	imull	%esi, %ecx
	subl	%r15d, %ecx
	movl	%ecx, 28(%rsp)
.L5:
	movslq	%r10d, %r11
.L4:
.LVL5:
.LBB5698:
.LBB5699:
	.loc 1 128 0
	movq	48(%rbx), %rdi
.LVL6:
	movq	%r11, %rsi
.LBB5700:
.LBB5701:
	.file 2 "/home/xiaocen/Software/opencv/include/opencv2/core/mat.hpp"
	.loc 2 430 0
	movl	76(%rbx), %r8d
.LBE5701:
.LBE5700:
.LBB5704:
	.loc 1 133 0
	xorl	%r9d, %r9d
.LBE5704:
.LBB5823:
.LBB5824:
	.loc 2 430 0
	movq	72(%rdi), %rcx
.LBE5824:
.LBE5823:
.LBB5825:
.LBB5702:
	addl	%r10d, %r8d
	movslq	%r8d, %r8
.LBE5702:
.LBE5825:
	.loc 1 128 0
	imulq	(%rcx), %rsi
	movslq	72(%rbx), %rcx
	leaq	(%rcx,%rcx,2), %rcx
	addq	%rcx, %rsi
	addq	16(%rdi), %rsi
.LVL7:
	.loc 1 129 0
	movq	56(%rbx), %rdi
.LVL8:
.LBB5826:
.LBB5703:
	.loc 2 430 0
	movq	72(%rdi), %rcx
	imulq	(%rcx), %r8
.LVL9:
.LBE5703:
.LBE5826:
.LBB5827:
	.loc 1 133 0
	movl	%edx, %ecx
	imull	%eax, %ecx
	addq	16(%rdi), %r8
.LVL10:
	testl	%ecx, %ecx
	jle	.L7
.LVL11:
	.p2align 4,,10
	.p2align 3
.L44:
.LBB5705:
.LBB5706:
.LBB5707:
	.file 3 "/usr/lib/gcc/x86_64-linux-gnu/5/include/emmintrin.h"
	.loc 3 698 0
	movdqu	(%rsi), %xmm0
.LBE5707:
.LBE5706:
.LBB5708:
.LBB5709:
.LBB5710:
.LBB5711:
	.file 4 "/usr/lib/gcc/x86_64-linux-gnu/5/include/xmmintrin.h"
	.loc 4 884 0
	movss	(%r12), %xmm6
.LVL12:
	leaq	4(%r12), %rcx
.LBE5711:
.LBE5710:
.LBE5709:
.LBE5708:
.LBB5712:
	.loc 1 149 0
	movq	16(%rbx), %rdi
	movl	$1, %edx
.LBE5712:
	.loc 1 138 0
	movdqa	(%rbx), %xmm5
.LBB5776:
.LBB5777:
	.loc 3 965 0
	movdqa	%xmm0, %xmm7
.LBE5777:
.LBE5776:
.LBB5779:
.LBB5780:
	.loc 4 743 0
	shufps	$0, %xmm6, %xmm6
.LVL13:
.LBE5780:
.LBE5779:
.LBB5781:
.LBB5778:
	.loc 3 965 0
	punpckhbw	%xmm5, %xmm7
.LVL14:
.LBE5778:
.LBE5781:
.LBB5782:
	.loc 1 149 0
	cmpl	$1, (%rdi)
.LBE5782:
.LBB5783:
.LBB5784:
	.loc 3 989 0
	punpcklbw	%xmm5, %xmm0
.LVL15:
.LBE5784:
.LBE5783:
.LBB5785:
.LBB5786:
	.loc 3 995 0
	movdqa	%xmm7, %xmm8
.LBE5786:
.LBE5785:
.LBB5788:
.LBB5789:
	.loc 3 971 0
	movdqa	%xmm0, %xmm9
.LBE5789:
.LBE5788:
.LBB5791:
.LBB5792:
	punpckhwd	%xmm5, %xmm7
.LVL16:
.LBE5792:
.LBE5791:
.LBB5793:
.LBB5794:
	.loc 3 995 0
	punpcklwd	%xmm5, %xmm0
.LVL17:
.LBE5794:
.LBE5793:
.LBB5795:
.LBB5790:
	.loc 3 971 0
	punpckhwd	%xmm5, %xmm9
.LVL18:
.LBE5790:
.LBE5795:
.LBB5796:
.LBB5787:
	.loc 3 995 0
	punpcklwd	%xmm5, %xmm8
.LVL19:
.LBE5787:
.LBE5796:
.LBB5797:
.LBB5798:
	.loc 3 767 0
	cvtdq2ps	%xmm7, %xmm7
.LVL20:
.LBE5798:
.LBE5797:
.LBB5799:
.LBB5800:
	cvtdq2ps	%xmm0, %xmm0
.LVL21:
.LBE5800:
.LBE5799:
.LBB5801:
.LBB5802:
	.loc 4 195 0
	mulps	%xmm6, %xmm7
.LVL22:
.LBE5802:
.LBE5801:
.LBB5803:
.LBB5804:
	.loc 3 767 0
	cvtdq2ps	%xmm9, %xmm9
.LVL23:
.LBE5804:
.LBE5803:
.LBB5805:
.LBB5806:
	cvtdq2ps	%xmm8, %xmm8
.LVL24:
.LBE5806:
.LBE5805:
.LBB5807:
.LBB5808:
	.loc 4 195 0
	mulps	%xmm6, %xmm9
.LBE5808:
.LBE5807:
.LBB5809:
.LBB5810:
	mulps	%xmm6, %xmm8
.LVL25:
.LBE5810:
.LBE5809:
.LBB5811:
.LBB5812:
	mulps	%xmm0, %xmm6
.LVL26:
.LBE5812:
.LBE5811:
.LBB5813:
	.loc 1 149 0
	jle	.L9
	movq	%r8, 8(%rsp)
.LVL27:
	.p2align 4,,10
	.p2align 3
.L45:
.LBB5713:
	.loc 1 152 0 discriminator 2
	movl	96(%rbx), %eax
.LBB5714:
.LBB5715:
	.loc 3 698 0 discriminator 2
	movq	%rsi, %r8
.LBE5715:
.LBE5714:
.LBB5718:
.LBB5719:
.LBB5720:
.LBB5721:
	.loc 4 884 0 discriminator 2
	movss	(%rcx), %xmm0
.LVL28:
	addq	$4, %rcx
.LBE5721:
.LBE5720:
.LBE5719:
.LBE5718:
.LBB5722:
.LBB5723:
	.loc 4 743 0 discriminator 2
	shufps	$0, %xmm0, %xmm0
.LVL29:
.LBE5723:
.LBE5722:
	.loc 1 152 0 discriminator 2
	imull	%edx, %eax
.LBE5713:
	.loc 1 149 0 discriminator 2
	addl	$1, %edx
.LVL30:
.LBB5774:
	.loc 1 152 0 discriminator 2
	cltq
.LVL31:
.LBB5724:
.LBB5716:
	.loc 3 698 0 discriminator 2
	subq	%rax, %r8
.LVL32:
.LBE5716:
.LBE5724:
.LBB5725:
.LBB5726:
	movdqu	(%rsi,%rax), %xmm3
.LVL33:
.LBE5726:
.LBE5725:
.LBE5774:
	.loc 1 149 0 discriminator 2
	cmpl	%edx, (%rdi)
.LBB5775:
.LBB5727:
.LBB5717:
	.loc 3 698 0 discriminator 2
	movdqu	(%r8), %xmm4
.LVL34:
.LBE5717:
.LBE5727:
.LBB5728:
.LBB5729:
	.loc 3 965 0 discriminator 2
	movdqa	%xmm3, %xmm1
.LBE5729:
.LBE5728:
.LBB5731:
.LBB5732:
	.loc 3 989 0 discriminator 2
	punpcklbw	%xmm5, %xmm3
.LVL35:
.LBE5732:
.LBE5731:
.LBB5733:
.LBB5734:
	.loc 3 965 0 discriminator 2
	movdqa	%xmm4, %xmm2
.LBE5734:
.LBE5733:
.LBB5736:
.LBB5730:
	punpckhbw	%xmm5, %xmm1
.LVL36:
.LBE5730:
.LBE5736:
.LBB5737:
.LBB5738:
	.loc 3 989 0 discriminator 2
	punpcklbw	%xmm5, %xmm4
.LVL37:
.LBE5738:
.LBE5737:
.LBB5739:
.LBB5735:
	.loc 3 965 0 discriminator 2
	punpckhbw	%xmm5, %xmm2
.LVL38:
.LBE5735:
.LBE5739:
.LBB5740:
.LBB5741:
	.loc 3 971 0 discriminator 2
	paddw	%xmm4, %xmm3
	movdqa	%xmm3, %xmm4
.LVL39:
.LBE5741:
.LBE5740:
.LBB5743:
.LBB5744:
	.loc 3 995 0 discriminator 2
	paddw	%xmm2, %xmm1
	movdqa	%xmm1, %xmm2
.LVL40:
.LBE5744:
.LBE5743:
.LBB5746:
.LBB5747:
	.loc 3 971 0 discriminator 2
	punpckhwd	%xmm5, %xmm1
.LBE5747:
.LBE5746:
.LBB5748:
.LBB5742:
	punpckhwd	%xmm5, %xmm4
.LVL41:
.LBE5742:
.LBE5748:
.LBB5749:
.LBB5745:
	.loc 3 995 0 discriminator 2
	punpcklwd	%xmm5, %xmm2
.LBE5745:
.LBE5749:
.LBB5750:
.LBB5751:
	.loc 3 767 0 discriminator 2
	cvtdq2ps	%xmm1, %xmm1
.LBE5751:
.LBE5750:
.LBB5752:
.LBB5753:
	.loc 4 183 0 discriminator 2
	mulps	%xmm0, %xmm1
.LBE5753:
.LBE5752:
.LBB5755:
.LBB5756:
	.loc 3 995 0 discriminator 2
	punpcklwd	%xmm5, %xmm3
.LVL42:
.LBE5756:
.LBE5755:
.LBB5757:
.LBB5754:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm7
.LVL43:
.LBE5754:
.LBE5757:
.LBB5758:
.LBB5759:
	.loc 3 767 0 discriminator 2
	cvtdq2ps	%xmm2, %xmm1
.LVL44:
.LBE5759:
.LBE5758:
.LBB5760:
.LBB5761:
	cvtdq2ps	%xmm3, %xmm3
.LVL45:
.LBE5761:
.LBE5760:
.LBB5762:
.LBB5763:
	.loc 4 183 0 discriminator 2
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm8
.LVL46:
.LBE5763:
.LBE5762:
.LBB5764:
.LBB5765:
	.loc 3 767 0 discriminator 2
	cvtdq2ps	%xmm4, %xmm1
.LVL47:
.LBE5765:
.LBE5764:
.LBB5766:
.LBB5767:
	.loc 4 183 0 discriminator 2
	mulps	%xmm0, %xmm1
.LBE5767:
.LBE5766:
.LBB5769:
.LBB5770:
	mulps	%xmm3, %xmm0
.LVL48:
.LBE5770:
.LBE5769:
.LBB5772:
.LBB5768:
	addps	%xmm1, %xmm9
.LVL49:
.LBE5768:
.LBE5772:
.LBB5773:
.LBB5771:
	addps	%xmm0, %xmm6
.LVL50:
.LBE5771:
.LBE5773:
.LBE5775:
	.loc 1 149 0 discriminator 2
	jg	.L45
	movq	8(%rsp), %r8
.LVL51:
.L9:
.LBE5813:
.LBB5814:
.LBB5815:
	.loc 4 980 0
	movups	%xmm6, (%r8)
.LVL52:
.LBE5815:
.LBE5814:
.LBE5705:
	.loc 1 133 0
	addl	$16, %r9d
.LVL53:
	addq	$16, %rsi
.LVL54:
	addq	$64, %r8
.LVL55:
.LBB5822:
.LBB5816:
.LBB5817:
	.loc 4 980 0
	movups	%xmm9, -48(%r8)
.LVL56:
.LBE5817:
.LBE5816:
.LBB5818:
.LBB5819:
	movups	%xmm8, -32(%r8)
.LVL57:
.LBE5819:
.LBE5818:
.LBB5820:
.LBB5821:
	movups	%xmm7, -16(%r8)
.LVL58:
.LBE5821:
.LBE5820:
.LBE5822:
	.loc 1 133 0
	movl	96(%rbx), %edx
	movl	84(%rbx), %eax
	movl	%edx, %ecx
	imull	%eax, %ecx
	cmpl	%r9d, %ecx
	jg	.L44
.LVL59:
.L7:
	addl	$1, %r10d
.LVL60:
	addq	$1, %r11
	cmpl	%r13d, %r10d
	jl	.L4
	movl	28(%rsp), %edi
	movl	%r15d, %r10d
.LVL61:
	leal	(%r15,%rdi), %r13d
	movl	24(%rsp), %edi
	cmpl	%r14d, %r13d
	cmovg	%r14d, %r13d
	addl	%edi, %r15d
	movl	%r15d, %ecx
	subl	%edi, %ecx
	cmpl	%ecx, %r14d
	jg	.L5
.LVL62:
.L2:
	call	GOMP_barrier
.LVL63:
.LBE5827:
.LBE5699:
.LBE5698:
.LBE5828:
.LBB5829:
	.loc 1 179 0
	movl	16(%rsp), %eax
	testl	%eax, %eax
	jne	.L30
.LBB5830:
	.loc 1 180 0
	movq	56(%rbx), %rdx
	.loc 1 181 0
	movslq	76(%rbx), %r14
	movl	92(%rbx), %eax
	movq	16(%rdx), %r15
.LVL64:
.LBB5831:
.LBB5832:
	.loc 2 430 0
	movq	72(%rdx), %rdx
.LVL65:
.LBE5832:
.LBE5831:
	.loc 1 181 0
	subl	%r14d, %eax
.LVL66:
	movq	%r14, %rcx
.LBB5837:
.LBB5833:
	.loc 2 430 0
	movslq	%eax, %r12
.LVL67:
.LBE5833:
.LBE5837:
.LBB5838:
.LBB5839:
	subl	$1, %eax
.LVL68:
.LBE5839:
.LBE5838:
.LBB5844:
.LBB5834:
	movq	(%rdx), %rdx
.LBE5834:
.LBE5844:
.LBB5845:
.LBB5840:
	cltq
.LBE5840:
.LBE5845:
.LBB5846:
.LBB5835:
	imulq	%rdx, %r12
.LBE5835:
.LBE5846:
.LBB5847:
.LBB5848:
	imulq	%rdx, %r14
.LBE5848:
.LBE5847:
.LBB5850:
.LBB5841:
	imulq	%rdx, %rax
.LVL69:
.LBE5841:
.LBE5850:
.LBB5851:
.LBB5836:
	addq	%r15, %r12
.LVL70:
.LBE5836:
.LBE5851:
.LBB5852:
.LBB5849:
	addq	%r15, %r14
.LVL71:
.LBE5849:
.LBE5852:
.LBB5853:
.LBB5842:
	addq	%r15, %rax
.LBE5842:
.LBE5853:
.LBB5854:
	.loc 1 184 0
	testl	%ecx, %ecx
.LBE5854:
.LBB5861:
.LBB5843:
	.loc 2 430 0
	movq	%rax, 8(%rsp)
.LVL72:
.LBE5843:
.LBE5861:
.LBB5862:
	.loc 1 184 0
	jle	.L30
	movslq	84(%rbx), %rax
	xorl	%r13d, %r13d
.LVL73:
.L31:
.LBB5855:
.LBB5856:
	.file 5 "/usr/include/x86_64-linux-gnu/bits/string3.h"
	.loc 5 53 0 discriminator 2
	leaq	(%rax,%rax,2), %rdx
	movq	%r15, %rdi
	movq	%r14, %rsi
.LBE5856:
.LBE5855:
	.loc 1 184 0 discriminator 2
	addl	$1, %r13d
.LVL74:
.LBB5858:
.LBB5857:
	.loc 5 53 0 discriminator 2
	salq	$2, %rdx
	call	memcpy
.LVL75:
.LBE5857:
.LBE5858:
.LBB5859:
.LBB5860:
	movslq	84(%rbx), %rax
	movq	8(%rsp), %rsi
	movq	%r12, %rdi
	leaq	(%rax,%rax,2), %rdx
	salq	$2, %rdx
	call	memcpy
.LVL76:
.LBE5860:
.LBE5859:
	.loc 1 184 0 discriminator 2
	movslq	84(%rbx), %rax
	leaq	(%rax,%rax,2), %rdx
	salq	$2, %rdx
	addq	%rdx, %r15
.LVL77:
	addq	%rdx, %r12
.LVL78:
	cmpl	%r13d, 76(%rbx)
	jg	.L31
.LVL79:
.L30:
.LBE5862:
.LBE5830:
.LBE5829:
.LBB5863:
	.loc 1 192 0
	movl	80(%rbx), %eax
	movl	16(%rsp), %r15d
	leal	3(%rax), %edx
	testl	%eax, %eax
	movl	%r15d, %r14d
	movl	%eax, %edi
	movl	%eax, 24(%rsp)
	cmovns	%eax, %edx
	sarl	$2, %edx
	imull	%edx, %r14d
	leal	(%rdx,%r14), %eax
	cmpl	%edi, %eax
	movl	%eax, %esi
	cmovg	%edi, %esi
	cmpl	%r14d, %edi
	movl	%esi, 8(%rsp)
	jle	.L12
	movl	20(%rsp), %esi
	movl	96(%rbx), %r13d
.LBB5864:
.LBB5865:
.LBB5866:
.LBB5867:
.LBB5868:
.LBB5869:
	.loc 5 53 0
	leaq	32(%rsp), %r12
	movl	84(%rbx), %eax
	movl	%esi, %edi
	movl	%r13d, %ecx
	imull	%edx, %edi
	movl	%edi, 28(%rsp)
	movl	%esi, %edi
	addl	%r15d, %edi
	movl	%edi, %esi
	addl	$1, %esi
	imull	%edx, %edi
	imull	%esi, %edx
	movl	%edi, 16(%rsp)
	subl	%edi, %edx
	movl	%edx, 20(%rsp)
.L15:
	movslq	%r14d, %r15
.L14:
.LVL80:
.LBE5869:
.LBE5868:
.LBE5867:
.LBE5866:
	.loc 1 195 0
	movq	56(%rbx), %rsi
.LVL81:
.LBB5974:
.LBB5975:
	.loc 2 430 0
	movl	76(%rbx), %edx
.LBE5975:
.LBE5974:
.LBB5977:
.LBB5978:
	movq	%r15, %r11
.LBE5978:
.LBE5977:
.LBB5980:
	.loc 1 200 0
	xorl	%r10d, %r10d
.LBE5980:
.LBB5981:
.LBB5976:
	.loc 2 430 0
	movq	72(%rsi), %rdi
	addl	%r14d, %edx
.LVL82:
	movslq	%edx, %rdx
	imulq	(%rdi), %rdx
.LVL83:
	addq	16(%rsi), %rdx
.LVL84:
.LBE5976:
.LBE5981:
	.loc 1 196 0
	movq	64(%rbx), %rsi
.LVL85:
.LBB5982:
.LBB5979:
	.loc 2 430 0
	movq	72(%rsi), %rdi
	imulq	(%rdi), %r11
.LVL86:
.LBE5979:
.LBE5982:
.LBB5983:
	.loc 1 200 0
	movl	%ecx, %edi
	imull	%eax, %edi
	addq	16(%rsi), %r11
.LVL87:
	testl	%edi, %edi
	jle	.L19
.LVL88:
	.p2align 4,,10
	.p2align 3
.L48:
.LBB5972:
.LBB5871:
	.loc 1 213 0
	movq	24(%rbx), %r8
.LBE5871:
.LBB5924:
.LBB5925:
.LBB5926:
.LBB5927:
	.loc 4 884 0
	movss	0(%rbp), %xmm3
.LVL89:
	leaq	4(%rbp), %rdi
.LBE5927:
.LBE5926:
.LBE5925:
.LBE5924:
.LBB5928:
.LBB5929:
	.loc 4 931 0
	movups	(%rdx), %xmm4
.LBE5929:
.LBE5928:
.LBB5930:
	.loc 1 213 0
	movl	$1, %esi
.LBE5930:
.LBB5931:
.LBB5932:
	.loc 4 743 0
	shufps	$0, %xmm3, %xmm3
.LVL90:
.LBE5932:
.LBE5931:
.LBB5933:
	.loc 1 213 0
	cmpl	$1, (%r8)
.LBE5933:
.LBB5934:
.LBB5935:
	.loc 4 931 0
	movups	16(%rdx), %xmm6
.LVL91:
.LBE5935:
.LBE5934:
.LBB5936:
.LBB5937:
	.loc 4 195 0
	mulps	%xmm3, %xmm4
.LVL92:
.LBE5937:
.LBE5936:
.LBB5938:
.LBB5939:
	.loc 4 931 0
	movups	32(%rdx), %xmm5
.LVL93:
.LBE5939:
.LBE5938:
.LBB5940:
.LBB5941:
	.loc 4 195 0
	mulps	%xmm3, %xmm6
.LVL94:
.LBE5941:
.LBE5940:
.LBB5942:
.LBB5943:
	.loc 4 931 0
	movups	48(%rdx), %xmm0
.LVL95:
.LBE5943:
.LBE5942:
.LBB5944:
.LBB5945:
	.loc 4 195 0
	mulps	%xmm3, %xmm5
.LVL96:
.LBE5945:
.LBE5944:
.LBB5946:
.LBB5947:
	mulps	%xmm0, %xmm3
.LVL97:
.LBE5947:
.LBE5946:
.LBB5948:
	.loc 1 213 0
	jle	.L28
.LVL98:
	.p2align 4,,10
	.p2align 3
.L46:
.LBB5872:
.LBB5873:
	.loc 4 931 0 discriminator 2
	movl	84(%rbx), %eax
.LBE5873:
.LBE5872:
.LBB5879:
.LBB5880:
.LBB5881:
.LBB5882:
	.loc 4 884 0 discriminator 2
	movss	(%rdi), %xmm0
.LVL99:
.LBE5882:
.LBE5881:
.LBE5880:
.LBE5879:
.LBB5883:
.LBB5874:
	.loc 4 931 0 discriminator 2
	movq	%rdx, %r9
	addq	$4, %rdi
.LBE5874:
.LBE5883:
.LBB5884:
.LBB5885:
	.loc 4 743 0 discriminator 2
	movaps	%xmm0, %xmm2
.LBE5885:
.LBE5884:
.LBB5887:
.LBB5875:
	.loc 4 931 0 discriminator 2
	imull	%esi, %eax
.LBE5875:
.LBE5887:
	.loc 1 213 0 discriminator 2
	addl	$1, %esi
.LVL100:
.LBB5888:
.LBB5876:
	.loc 4 931 0 discriminator 2
	imull	96(%rbx), %eax
.LBE5876:
.LBE5888:
.LBB5889:
.LBB5886:
	.loc 4 743 0 discriminator 2
	shufps	$0, %xmm0, %xmm2
.LVL101:
.LBE5886:
.LBE5889:
.LBB5890:
.LBB5877:
	.loc 4 931 0 discriminator 2
	cltq
	leaq	0(,%rax,4), %rcx
	subq	%rcx, %r9
.LBE5877:
.LBE5890:
.LBB5891:
.LBB5892:
	movups	(%rdx,%rcx), %xmm0
.LBE5892:
.LBE5891:
.LBB5893:
.LBB5894:
	movq	%rax, %rcx
.LBE5894:
.LBE5893:
.LBB5897:
.LBB5878:
	movups	(%r9), %xmm1
.LVL102:
.LBE5878:
.LBE5897:
.LBB5898:
.LBB5895:
	negq	%rcx
	salq	$2, %rcx
.LBE5895:
.LBE5898:
	.loc 1 213 0 discriminator 2
	cmpl	%esi, (%r8)
.LBB5899:
.LBB5900:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm0
.LBE5900:
.LBE5899:
.LBB5902:
.LBB5896:
	.loc 4 931 0 discriminator 2
	movups	16(%rdx,%rcx), %xmm1
.LBE5896:
.LBE5902:
.LBB5903:
.LBB5901:
	.loc 4 183 0 discriminator 2
	mulps	%xmm2, %xmm0
	addps	%xmm0, %xmm4
.LVL103:
.LBE5901:
.LBE5903:
.LBB5904:
.LBB5905:
	.loc 4 931 0 discriminator 2
	movups	16(%rdx,%rax,4), %xmm0
.LVL104:
.LBE5905:
.LBE5904:
.LBB5906:
.LBB5907:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm0
.LBE5907:
.LBE5906:
.LBB5909:
.LBB5910:
	.loc 4 931 0 discriminator 2
	movups	32(%rdx,%rcx), %xmm1
.LBE5910:
.LBE5909:
.LBB5911:
.LBB5908:
	.loc 4 183 0 discriminator 2
	mulps	%xmm2, %xmm0
	addps	%xmm0, %xmm6
.LVL105:
.LBE5908:
.LBE5911:
.LBB5912:
.LBB5913:
	.loc 4 931 0 discriminator 2
	movups	32(%rdx,%rax,4), %xmm0
.LVL106:
.LBE5913:
.LBE5912:
.LBB5914:
.LBB5915:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm0
.LBE5915:
.LBE5914:
.LBB5917:
.LBB5918:
	.loc 4 931 0 discriminator 2
	movups	48(%rdx,%rax,4), %xmm1
.LBE5918:
.LBE5917:
.LBB5919:
.LBB5916:
	.loc 4 183 0 discriminator 2
	mulps	%xmm2, %xmm0
	addps	%xmm0, %xmm5
.LVL107:
.LBE5916:
.LBE5919:
.LBB5920:
.LBB5921:
	.loc 4 931 0 discriminator 2
	movups	48(%rdx,%rcx), %xmm0
.LVL108:
.LBE5921:
.LBE5920:
.LBB5922:
.LBB5923:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm0
	mulps	%xmm2, %xmm0
	addps	%xmm0, %xmm3
.LVL109:
.LBE5923:
.LBE5922:
	.loc 1 213 0 discriminator 2
	jg	.L46
.LVL110:
.L28:
.LBE5948:
	.loc 1 233 0
	movl	96(%rbx), %ecx
	movl	88(%rbx), %eax
.LBB5949:
.LBB5950:
	.loc 3 815 0
	cvttps2dq	%xmm4, %xmm4
.LVL111:
.LBE5950:
.LBE5949:
.LBB5951:
.LBB5952:
	cvttps2dq	%xmm6, %xmm6
.LVL112:
.LBE5952:
.LBE5951:
.LBB5953:
.LBB5954:
	cvttps2dq	%xmm5, %xmm5
.LVL113:
.LBE5954:
.LBE5953:
.LBB5955:
.LBB5956:
	cvttps2dq	%xmm3, %xmm3
.LVL114:
.LBE5956:
.LBE5955:
.LBB5957:
.LBB5958:
	.loc 3 953 0
	packssdw	%xmm6, %xmm4
.LVL115:
.LBE5958:
.LBE5957:
.LBB5959:
.LBB5960:
	packssdw	%xmm3, %xmm5
.LVL116:
.LBE5960:
.LBE5959:
	.loc 1 233 0
	imull	%ecx, %eax
.LBB5961:
.LBB5962:
	.loc 3 959 0
	packuswb	%xmm5, %xmm4
.LVL117:
.LBE5962:
.LBE5961:
	.loc 1 233 0
	subl	%r10d, %eax
.LVL118:
.LBB5963:
.LBB5964:
	.file 6 "/usr/include/c++/5/bits/stl_algobase.h"
	.loc 6 200 0
	cmpl	$15, %eax
.LBE5964:
.LBE5963:
.LBB5966:
.LBB5967:
	.loc 3 710 0
	movaps	%xmm4, 32(%rsp)
.LBE5967:
.LBE5966:
.LBB5969:
.LBB5965:
	.loc 6 200 0
	jg	.L17
.LVL119:
.LBE5965:
.LBE5969:
	.loc 1 235 0
	testl	%eax, %eax
	jg	.L18
.LVL120:
.L27:
.LBE5972:
	.loc 1 200 0
	movl	84(%rbx), %eax
	addl	$16, %r10d
.LVL121:
	addq	$64, %rdx
.LVL122:
	addq	$16, %r11
	movl	%eax, %esi
	imull	%ecx, %esi
	cmpl	%r10d, %esi
	jg	.L48
.LVL123:
.L19:
	addl	$1, %r14d
.LVL124:
	addq	$1, %r15
	cmpl	8(%rsp), %r14d
	jl	.L14
	movl	16(%rsp), %edi
	movl	20(%rsp), %edx
.LVL125:
	movl	24(%rsp), %esi
	addl	%edi, %edx
	movl	%edi, %r14d
.LVL126:
	cmpl	%esi, %edx
	cmovg	%esi, %edx
	movl	%edx, 8(%rsp)
	movl	%edi, %edx
	movl	28(%rsp), %edi
	addl	%edi, %edx
	movl	%edx, 16(%rsp)
	subl	%edi, %edx
	cmpl	%edx, %esi
	jg	.L15
.L12:
	call	GOMP_barrier
.LVL127:
.LBE5983:
.LBE5865:
.LBE5864:
.LBE5863:
	.loc 1 120 0
	movq	56(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L60
	addq	$72, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
.LVL128:
	popq	%rbp
	.cfi_def_cfa_offset 40
.LVL129:
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.LVL130:
	.p2align 4,,10
	.p2align 3
.L17:
	.cfi_restore_state
.LBB5987:
.LBB5986:
.LBB5985:
.LBB5984:
.LBB5973:
.LBB5970:
.LBB5968:
	.loc 3 710 0
	movl	$16, %eax
.LVL131:
.L32:
.LBE5968:
.LBE5970:
.LBB5971:
.LBB5870:
	.loc 5 53 0 discriminator 1
	cmpl	$8, %eax
	movl	%eax, %ecx
	jnb	.L21
	testb	$4, %al
	jne	.L61
	testl	%ecx, %ecx
	je	.L22
	movzbl	(%r12), %eax
	testb	$2, %cl
	movb	%al, (%r11)
	jne	.L62
.L57:
	movl	96(%rbx), %r13d
.L22:
	movl	%r13d, %ecx
	jmp	.L27
	.p2align 4,,10
	.p2align 3
.L21:
	movq	(%r12), %rcx
	movq	%r12, %rdi
	movq	%rcx, (%r11)
	movl	%eax, %ecx
	movq	-8(%r12,%rcx), %rsi
	movq	%rsi, -8(%r11,%rcx)
	leaq	8(%r11), %rsi
	movq	%r11, %rcx
	andq	$-8, %rsi
	subq	%rsi, %rcx
	subq	%rcx, %rdi
	addl	%eax, %ecx
	andl	$-8, %ecx
	cmpl	$8, %ecx
	jb	.L57
	andl	$-8, %ecx
	xorl	%eax, %eax
.L25:
	movl	%eax, %r8d
	addl	$8, %eax
	movq	(%rdi,%r8), %r9
	cmpl	%ecx, %eax
	movq	%r9, (%rsi,%r8)
	jb	.L25
	jmp	.L57
.L61:
	movl	(%r12), %eax
	movl	%eax, (%r11)
	movl	-4(%r12,%rcx), %eax
	movl	%eax, -4(%r11,%rcx)
	movl	96(%rbx), %r13d
	jmp	.L22
.LVL132:
.L18:
	cltq
	jmp	.L32
.LVL133:
.L62:
	movzwl	-2(%r12,%rcx), %eax
	movw	%ax, -2(%r11,%rcx)
	movl	96(%rbx), %r13d
	jmp	.L22
.LVL134:
.L60:
.LBE5870:
.LBE5971:
.LBE5973:
.LBE5984:
.LBE5985:
.LBE5986:
.LBE5987:
	.loc 1 120 0
	call	__stack_chk_fail
.LVL135:
	.cfi_endproc
.LFE12399:
	.size	_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.0, .-_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.0
	.section	.text.unlikely._Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.0,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LCOLDE0:
	.section	.text._Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.0,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LHOTE0:
	.section	.text.unlikely._Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.1,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LCOLDB1:
	.section	.text._Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.1,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LHOTB1:
	.p2align 4,,15
	.type	_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.1, @function
_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.1:
.LFB12400:
	.loc 1 120 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
.LVL136:
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rdi, %rbx
	subq	$72, %rsp
	.cfi_def_cfa_offset 128
.LBB5988:
	.loc 1 125 0
	movl	80(%rdi), %r14d
.LBE5988:
	.loc 1 120 0
	movq	40(%rdi), %r12
.LVL137:
	movq	%fs:40, %rax
	movq	%rax, 56(%rsp)
	xorl	%eax, %eax
	movq	32(%rdi), %rbp
.LVL138:
.LBB6119:
	.loc 1 125 0
	leal	3(%r14), %ecx
	testl	%r14d, %r14d
	cmovns	%r14d, %ecx
	sarl	$2, %ecx
	movl	%ecx, 8(%rsp)
	call	omp_get_num_threads
.LVL139:
	movl	%eax, %r15d
	movl	%eax, 20(%rsp)
	call	omp_get_thread_num
.LVL140:
	movl	8(%rsp), %ecx
	movl	%eax, %r10d
	movl	%eax, 16(%rsp)
	imull	%ecx, %r10d
	leal	(%rcx,%r10), %r13d
	cmpl	%r14d, %r13d
	cmovg	%r14d, %r13d
	cmpl	%r10d, %r14d
	jle	.L64
	movl	%r15d, %edi
	movl	%eax, %r11d
	movl	96(%rbx), %edx
	imull	%ecx, %edi
	movl	84(%rbx), %eax
	movl	%edi, 24(%rsp)
	movl	%r15d, %edi
	addl	%r11d, %edi
	movl	%edi, %esi
	movl	%edi, %r15d
	addl	$1, %esi
	imull	%ecx, %r15d
	imull	%esi, %ecx
	subl	%r15d, %ecx
	movl	%ecx, 28(%rsp)
.L67:
	movslq	%r10d, %r11
.L66:
.LVL141:
.LBB5989:
.LBB5990:
	.loc 1 128 0
	movq	48(%rbx), %rcx
.LVL142:
	movq	%r11, %rdi
.LBB5991:
.LBB5992:
	.loc 2 430 0
	movl	76(%rbx), %r8d
.LBE5992:
.LBE5991:
.LBB5995:
	.loc 1 133 0
	xorl	%r9d, %r9d
.LBE5995:
.LBB6114:
.LBB6115:
	.loc 2 430 0
	movq	72(%rcx), %rsi
.LBE6115:
.LBE6114:
.LBB6116:
.LBB5993:
	addl	%r10d, %r8d
	movslq	%r8d, %r8
.LBE5993:
.LBE6116:
	.loc 1 128 0
	imulq	(%rsi), %rdi
	movq	%rdi, %rsi
	movslq	72(%rbx), %rdi
	leaq	(%rsi,%rdi,2), %rsi
	.loc 1 129 0
	movq	56(%rbx), %rdi
	.loc 1 128 0
	addq	16(%rcx), %rsi
.LVL143:
.LBB6117:
.LBB5994:
	.loc 2 430 0
	movq	72(%rdi), %rcx
.LVL144:
	imulq	(%rcx), %r8
.LVL145:
.LBE5994:
.LBE6117:
.LBB6118:
	.loc 1 133 0
	movl	%edx, %ecx
	imull	%eax, %ecx
	addq	16(%rdi), %r8
.LVL146:
	testl	%ecx, %ecx
	jle	.L69
.LVL147:
	.p2align 4,,10
	.p2align 3
.L106:
.LBB5996:
.LBB5997:
.LBB5998:
	.loc 3 698 0
	movdqu	(%rsi), %xmm0
.LBE5998:
.LBE5997:
.LBB5999:
.LBB6000:
.LBB6001:
.LBB6002:
	.loc 4 884 0
	movss	(%r12), %xmm6
.LVL148:
	leaq	4(%r12), %rcx
.LBE6002:
.LBE6001:
.LBE6000:
.LBE5999:
.LBB6003:
	.loc 1 149 0
	movq	16(%rbx), %rdi
	movl	$1, %edx
.LBE6003:
	.loc 1 138 0
	movdqa	(%rbx), %xmm5
.LBB6067:
.LBB6068:
	.loc 3 965 0
	movdqa	%xmm0, %xmm7
.LBE6068:
.LBE6067:
.LBB6070:
.LBB6071:
	.loc 4 743 0
	shufps	$0, %xmm6, %xmm6
.LVL149:
.LBE6071:
.LBE6070:
.LBB6072:
.LBB6069:
	.loc 3 965 0
	punpckhbw	%xmm5, %xmm7
.LVL150:
.LBE6069:
.LBE6072:
.LBB6073:
	.loc 1 149 0
	cmpl	$1, (%rdi)
.LBE6073:
.LBB6074:
.LBB6075:
	.loc 3 989 0
	punpcklbw	%xmm5, %xmm0
.LVL151:
.LBE6075:
.LBE6074:
.LBB6076:
.LBB6077:
	.loc 3 995 0
	movdqa	%xmm7, %xmm8
.LBE6077:
.LBE6076:
.LBB6079:
.LBB6080:
	.loc 3 971 0
	movdqa	%xmm0, %xmm9
.LBE6080:
.LBE6079:
.LBB6082:
.LBB6083:
	punpckhwd	%xmm5, %xmm7
.LVL152:
.LBE6083:
.LBE6082:
.LBB6084:
.LBB6085:
	.loc 3 995 0
	punpcklwd	%xmm5, %xmm0
.LVL153:
.LBE6085:
.LBE6084:
.LBB6086:
.LBB6081:
	.loc 3 971 0
	punpckhwd	%xmm5, %xmm9
.LVL154:
.LBE6081:
.LBE6086:
.LBB6087:
.LBB6078:
	.loc 3 995 0
	punpcklwd	%xmm5, %xmm8
.LVL155:
.LBE6078:
.LBE6087:
.LBB6088:
.LBB6089:
	.loc 3 767 0
	cvtdq2ps	%xmm7, %xmm7
.LVL156:
.LBE6089:
.LBE6088:
.LBB6090:
.LBB6091:
	cvtdq2ps	%xmm0, %xmm0
.LVL157:
.LBE6091:
.LBE6090:
.LBB6092:
.LBB6093:
	.loc 4 195 0
	mulps	%xmm6, %xmm7
.LVL158:
.LBE6093:
.LBE6092:
.LBB6094:
.LBB6095:
	.loc 3 767 0
	cvtdq2ps	%xmm9, %xmm9
.LVL159:
.LBE6095:
.LBE6094:
.LBB6096:
.LBB6097:
	cvtdq2ps	%xmm8, %xmm8
.LVL160:
.LBE6097:
.LBE6096:
.LBB6098:
.LBB6099:
	.loc 4 195 0
	mulps	%xmm6, %xmm9
.LBE6099:
.LBE6098:
.LBB6100:
.LBB6101:
	mulps	%xmm6, %xmm8
.LVL161:
.LBE6101:
.LBE6100:
.LBB6102:
.LBB6103:
	mulps	%xmm0, %xmm6
.LVL162:
.LBE6103:
.LBE6102:
.LBB6104:
	.loc 1 149 0
	jle	.L71
	movq	%r8, 8(%rsp)
.LVL163:
	.p2align 4,,10
	.p2align 3
.L107:
.LBB6004:
	.loc 1 152 0 discriminator 2
	movl	96(%rbx), %eax
.LBB6005:
.LBB6006:
	.loc 3 698 0 discriminator 2
	movq	%rsi, %r8
.LBE6006:
.LBE6005:
.LBB6009:
.LBB6010:
.LBB6011:
.LBB6012:
	.loc 4 884 0 discriminator 2
	movss	(%rcx), %xmm0
.LVL164:
	addq	$4, %rcx
.LBE6012:
.LBE6011:
.LBE6010:
.LBE6009:
.LBB6013:
.LBB6014:
	.loc 4 743 0 discriminator 2
	shufps	$0, %xmm0, %xmm0
.LVL165:
.LBE6014:
.LBE6013:
	.loc 1 152 0 discriminator 2
	imull	%edx, %eax
.LBE6004:
	.loc 1 149 0 discriminator 2
	addl	$1, %edx
.LVL166:
.LBB6065:
	.loc 1 152 0 discriminator 2
	cltq
.LVL167:
.LBB6015:
.LBB6007:
	.loc 3 698 0 discriminator 2
	subq	%rax, %r8
.LVL168:
.LBE6007:
.LBE6015:
.LBB6016:
.LBB6017:
	movdqu	(%rsi,%rax), %xmm3
.LVL169:
.LBE6017:
.LBE6016:
.LBE6065:
	.loc 1 149 0 discriminator 2
	cmpl	%edx, (%rdi)
.LBB6066:
.LBB6018:
.LBB6008:
	.loc 3 698 0 discriminator 2
	movdqu	(%r8), %xmm4
.LVL170:
.LBE6008:
.LBE6018:
.LBB6019:
.LBB6020:
	.loc 3 965 0 discriminator 2
	movdqa	%xmm3, %xmm1
.LBE6020:
.LBE6019:
.LBB6022:
.LBB6023:
	.loc 3 989 0 discriminator 2
	punpcklbw	%xmm5, %xmm3
.LVL171:
.LBE6023:
.LBE6022:
.LBB6024:
.LBB6025:
	.loc 3 965 0 discriminator 2
	movdqa	%xmm4, %xmm2
.LBE6025:
.LBE6024:
.LBB6027:
.LBB6021:
	punpckhbw	%xmm5, %xmm1
.LVL172:
.LBE6021:
.LBE6027:
.LBB6028:
.LBB6029:
	.loc 3 989 0 discriminator 2
	punpcklbw	%xmm5, %xmm4
.LVL173:
.LBE6029:
.LBE6028:
.LBB6030:
.LBB6026:
	.loc 3 965 0 discriminator 2
	punpckhbw	%xmm5, %xmm2
.LVL174:
.LBE6026:
.LBE6030:
.LBB6031:
.LBB6032:
	.loc 3 971 0 discriminator 2
	paddw	%xmm4, %xmm3
	movdqa	%xmm3, %xmm4
.LVL175:
.LBE6032:
.LBE6031:
.LBB6034:
.LBB6035:
	.loc 3 995 0 discriminator 2
	paddw	%xmm2, %xmm1
	movdqa	%xmm1, %xmm2
.LVL176:
.LBE6035:
.LBE6034:
.LBB6037:
.LBB6038:
	.loc 3 971 0 discriminator 2
	punpckhwd	%xmm5, %xmm1
.LBE6038:
.LBE6037:
.LBB6039:
.LBB6033:
	punpckhwd	%xmm5, %xmm4
.LVL177:
.LBE6033:
.LBE6039:
.LBB6040:
.LBB6036:
	.loc 3 995 0 discriminator 2
	punpcklwd	%xmm5, %xmm2
.LBE6036:
.LBE6040:
.LBB6041:
.LBB6042:
	.loc 3 767 0 discriminator 2
	cvtdq2ps	%xmm1, %xmm1
.LBE6042:
.LBE6041:
.LBB6043:
.LBB6044:
	.loc 4 183 0 discriminator 2
	mulps	%xmm0, %xmm1
.LBE6044:
.LBE6043:
.LBB6046:
.LBB6047:
	.loc 3 995 0 discriminator 2
	punpcklwd	%xmm5, %xmm3
.LVL178:
.LBE6047:
.LBE6046:
.LBB6048:
.LBB6045:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm7
.LVL179:
.LBE6045:
.LBE6048:
.LBB6049:
.LBB6050:
	.loc 3 767 0 discriminator 2
	cvtdq2ps	%xmm2, %xmm1
.LVL180:
.LBE6050:
.LBE6049:
.LBB6051:
.LBB6052:
	cvtdq2ps	%xmm3, %xmm3
.LVL181:
.LBE6052:
.LBE6051:
.LBB6053:
.LBB6054:
	.loc 4 183 0 discriminator 2
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm8
.LVL182:
.LBE6054:
.LBE6053:
.LBB6055:
.LBB6056:
	.loc 3 767 0 discriminator 2
	cvtdq2ps	%xmm4, %xmm1
.LVL183:
.LBE6056:
.LBE6055:
.LBB6057:
.LBB6058:
	.loc 4 183 0 discriminator 2
	mulps	%xmm0, %xmm1
.LBE6058:
.LBE6057:
.LBB6060:
.LBB6061:
	mulps	%xmm3, %xmm0
.LVL184:
.LBE6061:
.LBE6060:
.LBB6063:
.LBB6059:
	addps	%xmm1, %xmm9
.LVL185:
.LBE6059:
.LBE6063:
.LBB6064:
.LBB6062:
	addps	%xmm0, %xmm6
.LVL186:
.LBE6062:
.LBE6064:
.LBE6066:
	.loc 1 149 0 discriminator 2
	jg	.L107
	movq	8(%rsp), %r8
.LVL187:
.L71:
.LBE6104:
.LBB6105:
.LBB6106:
	.loc 4 980 0
	movups	%xmm6, (%r8)
.LVL188:
.LBE6106:
.LBE6105:
.LBE5996:
	.loc 1 133 0
	addl	$16, %r9d
.LVL189:
	addq	$16, %rsi
.LVL190:
	addq	$64, %r8
.LVL191:
.LBB6113:
.LBB6107:
.LBB6108:
	.loc 4 980 0
	movups	%xmm9, -48(%r8)
.LVL192:
.LBE6108:
.LBE6107:
.LBB6109:
.LBB6110:
	movups	%xmm8, -32(%r8)
.LVL193:
.LBE6110:
.LBE6109:
.LBB6111:
.LBB6112:
	movups	%xmm7, -16(%r8)
.LVL194:
.LBE6112:
.LBE6111:
.LBE6113:
	.loc 1 133 0
	movl	96(%rbx), %edx
	movl	84(%rbx), %eax
	movl	%edx, %ecx
	imull	%eax, %ecx
	cmpl	%r9d, %ecx
	jg	.L106
.LVL195:
.L69:
	addl	$1, %r10d
.LVL196:
	addq	$1, %r11
	cmpl	%r13d, %r10d
	jl	.L66
	movl	28(%rsp), %edi
	movl	%r15d, %r10d
.LVL197:
	leal	(%r15,%rdi), %r13d
	movl	24(%rsp), %edi
	cmpl	%r14d, %r13d
	cmovg	%r14d, %r13d
	addl	%edi, %r15d
	movl	%r15d, %ecx
	subl	%edi, %ecx
	cmpl	%ecx, %r14d
	jg	.L67
.LVL198:
.L64:
	call	GOMP_barrier
.LVL199:
.LBE6118:
.LBE5990:
.LBE5989:
.LBE6119:
.LBB6120:
	.loc 1 179 0
	movl	16(%rsp), %eax
	testl	%eax, %eax
	jne	.L92
.LBB6121:
	.loc 1 180 0
	movq	56(%rbx), %rdx
	.loc 1 181 0
	movslq	76(%rbx), %r14
	movl	92(%rbx), %eax
	movq	16(%rdx), %r15
.LVL200:
.LBB6122:
.LBB6123:
	.loc 2 430 0
	movq	72(%rdx), %rdx
.LVL201:
.LBE6123:
.LBE6122:
	.loc 1 181 0
	subl	%r14d, %eax
.LVL202:
	movq	%r14, %rcx
.LBB6128:
.LBB6124:
	.loc 2 430 0
	movslq	%eax, %r12
.LVL203:
.LBE6124:
.LBE6128:
.LBB6129:
.LBB6130:
	subl	$1, %eax
.LVL204:
.LBE6130:
.LBE6129:
.LBB6135:
.LBB6125:
	movq	(%rdx), %rdx
.LBE6125:
.LBE6135:
.LBB6136:
.LBB6131:
	cltq
.LBE6131:
.LBE6136:
.LBB6137:
.LBB6126:
	imulq	%rdx, %r12
.LBE6126:
.LBE6137:
.LBB6138:
.LBB6139:
	imulq	%rdx, %r14
.LBE6139:
.LBE6138:
.LBB6141:
.LBB6132:
	imulq	%rdx, %rax
.LVL205:
.LBE6132:
.LBE6141:
.LBB6142:
.LBB6127:
	addq	%r15, %r12
.LVL206:
.LBE6127:
.LBE6142:
.LBB6143:
.LBB6140:
	addq	%r15, %r14
.LVL207:
.LBE6140:
.LBE6143:
.LBB6144:
.LBB6133:
	addq	%r15, %rax
.LBE6133:
.LBE6144:
.LBB6145:
	.loc 1 184 0
	testl	%ecx, %ecx
.LBE6145:
.LBB6152:
.LBB6134:
	.loc 2 430 0
	movq	%rax, 8(%rsp)
.LVL208:
.LBE6134:
.LBE6152:
.LBB6153:
	.loc 1 184 0
	jle	.L92
	movslq	84(%rbx), %rax
	xorl	%r13d, %r13d
.LVL209:
.L93:
.LBB6146:
.LBB6147:
	.loc 5 53 0 discriminator 2
	leaq	0(,%rax,8), %rdx
	movq	%r15, %rdi
	movq	%r14, %rsi
.LBE6147:
.LBE6146:
	.loc 1 184 0 discriminator 2
	addl	$1, %r13d
.LVL210:
.LBB6149:
.LBB6148:
	.loc 5 53 0 discriminator 2
	call	memcpy
.LVL211:
.LBE6148:
.LBE6149:
.LBB6150:
.LBB6151:
	movslq	84(%rbx), %rdx
	movq	8(%rsp), %rsi
	movq	%r12, %rdi
	salq	$3, %rdx
.LVL212:
	call	memcpy
.LVL213:
.LBE6151:
.LBE6150:
	.loc 1 184 0 discriminator 2
	movslq	84(%rbx), %rax
	leaq	0(,%rax,8), %rdx
	addq	%rdx, %r15
.LVL214:
	addq	%rdx, %r12
.LVL215:
	cmpl	%r13d, 76(%rbx)
	jg	.L93
.LVL216:
.L92:
.LBE6153:
.LBE6121:
.LBE6120:
.LBB6154:
	.loc 1 192 0
	movl	80(%rbx), %eax
	movl	16(%rsp), %r15d
	leal	3(%rax), %edx
	testl	%eax, %eax
	movl	%r15d, %r14d
	movl	%eax, %edi
	movl	%eax, 24(%rsp)
	cmovns	%eax, %edx
	sarl	$2, %edx
	imull	%edx, %r14d
	leal	(%rdx,%r14), %eax
	cmpl	%edi, %eax
	movl	%eax, %esi
	cmovg	%edi, %esi
	cmpl	%r14d, %edi
	movl	%esi, 8(%rsp)
	jle	.L74
	movl	20(%rsp), %esi
	movl	96(%rbx), %r13d
.LBB6155:
.LBB6156:
.LBB6157:
.LBB6158:
.LBB6159:
.LBB6160:
	.loc 5 53 0
	leaq	32(%rsp), %r12
	movl	84(%rbx), %eax
	movl	%esi, %edi
	movl	%r13d, %ecx
	imull	%edx, %edi
	movl	%edi, 28(%rsp)
	movl	%esi, %edi
	addl	%r15d, %edi
	movl	%edi, %esi
	addl	$1, %esi
	imull	%edx, %edi
	imull	%esi, %edx
	movl	%edi, 16(%rsp)
	subl	%edi, %edx
	movl	%edx, 20(%rsp)
.L77:
	movslq	%r14d, %r15
.L76:
.LVL217:
.LBE6160:
.LBE6159:
.LBE6158:
.LBE6157:
	.loc 1 195 0
	movq	56(%rbx), %rsi
.LVL218:
.LBB6265:
.LBB6266:
	.loc 2 430 0
	movl	76(%rbx), %edx
.LBE6266:
.LBE6265:
.LBB6268:
.LBB6269:
	movq	%r15, %r11
.LBE6269:
.LBE6268:
.LBB6271:
	.loc 1 200 0
	xorl	%r10d, %r10d
.LBE6271:
.LBB6272:
.LBB6267:
	.loc 2 430 0
	movq	72(%rsi), %rdi
	addl	%r14d, %edx
.LVL219:
	movslq	%edx, %rdx
	imulq	(%rdi), %rdx
.LVL220:
	addq	16(%rsi), %rdx
.LVL221:
.LBE6267:
.LBE6272:
	.loc 1 196 0
	movq	64(%rbx), %rsi
.LVL222:
.LBB6273:
.LBB6270:
	.loc 2 430 0
	movq	72(%rsi), %rdi
	imulq	(%rdi), %r11
.LVL223:
.LBE6270:
.LBE6273:
.LBB6274:
	.loc 1 200 0
	movl	%ecx, %edi
	imull	%eax, %edi
	addq	16(%rsi), %r11
.LVL224:
	testl	%edi, %edi
	jle	.L81
.LVL225:
	.p2align 4,,10
	.p2align 3
.L110:
.LBB6263:
.LBB6162:
	.loc 1 213 0
	movq	24(%rbx), %r8
.LBE6162:
.LBB6215:
.LBB6216:
.LBB6217:
.LBB6218:
	.loc 4 884 0
	movss	0(%rbp), %xmm3
.LVL226:
	leaq	4(%rbp), %rdi
.LBE6218:
.LBE6217:
.LBE6216:
.LBE6215:
.LBB6219:
.LBB6220:
	.loc 4 931 0
	movups	(%rdx), %xmm4
.LBE6220:
.LBE6219:
.LBB6221:
	.loc 1 213 0
	movl	$1, %esi
.LBE6221:
.LBB6222:
.LBB6223:
	.loc 4 743 0
	shufps	$0, %xmm3, %xmm3
.LVL227:
.LBE6223:
.LBE6222:
.LBB6224:
	.loc 1 213 0
	cmpl	$1, (%r8)
.LBE6224:
.LBB6225:
.LBB6226:
	.loc 4 931 0
	movups	16(%rdx), %xmm6
.LVL228:
.LBE6226:
.LBE6225:
.LBB6227:
.LBB6228:
	.loc 4 195 0
	mulps	%xmm3, %xmm4
.LVL229:
.LBE6228:
.LBE6227:
.LBB6229:
.LBB6230:
	.loc 4 931 0
	movups	32(%rdx), %xmm5
.LVL230:
.LBE6230:
.LBE6229:
.LBB6231:
.LBB6232:
	.loc 4 195 0
	mulps	%xmm3, %xmm6
.LVL231:
.LBE6232:
.LBE6231:
.LBB6233:
.LBB6234:
	.loc 4 931 0
	movups	48(%rdx), %xmm0
.LVL232:
.LBE6234:
.LBE6233:
.LBB6235:
.LBB6236:
	.loc 4 195 0
	mulps	%xmm3, %xmm5
.LVL233:
.LBE6236:
.LBE6235:
.LBB6237:
.LBB6238:
	mulps	%xmm0, %xmm3
.LVL234:
.LBE6238:
.LBE6237:
.LBB6239:
	.loc 1 213 0
	jle	.L90
.LVL235:
	.p2align 4,,10
	.p2align 3
.L108:
.LBB6163:
.LBB6164:
	.loc 4 931 0 discriminator 2
	movl	84(%rbx), %eax
.LBE6164:
.LBE6163:
.LBB6170:
.LBB6171:
.LBB6172:
.LBB6173:
	.loc 4 884 0 discriminator 2
	movss	(%rdi), %xmm0
.LVL236:
.LBE6173:
.LBE6172:
.LBE6171:
.LBE6170:
.LBB6174:
.LBB6165:
	.loc 4 931 0 discriminator 2
	movq	%rdx, %r9
	addq	$4, %rdi
.LBE6165:
.LBE6174:
.LBB6175:
.LBB6176:
	.loc 4 743 0 discriminator 2
	movaps	%xmm0, %xmm2
.LBE6176:
.LBE6175:
.LBB6178:
.LBB6166:
	.loc 4 931 0 discriminator 2
	imull	%esi, %eax
.LBE6166:
.LBE6178:
	.loc 1 213 0 discriminator 2
	addl	$1, %esi
.LVL237:
.LBB6179:
.LBB6167:
	.loc 4 931 0 discriminator 2
	imull	96(%rbx), %eax
.LBE6167:
.LBE6179:
.LBB6180:
.LBB6177:
	.loc 4 743 0 discriminator 2
	shufps	$0, %xmm0, %xmm2
.LVL238:
.LBE6177:
.LBE6180:
.LBB6181:
.LBB6168:
	.loc 4 931 0 discriminator 2
	cltq
	leaq	0(,%rax,4), %rcx
	subq	%rcx, %r9
.LBE6168:
.LBE6181:
.LBB6182:
.LBB6183:
	movups	(%rdx,%rcx), %xmm0
.LBE6183:
.LBE6182:
.LBB6184:
.LBB6185:
	movq	%rax, %rcx
.LBE6185:
.LBE6184:
.LBB6188:
.LBB6169:
	movups	(%r9), %xmm1
.LVL239:
.LBE6169:
.LBE6188:
.LBB6189:
.LBB6186:
	negq	%rcx
	salq	$2, %rcx
.LBE6186:
.LBE6189:
	.loc 1 213 0 discriminator 2
	cmpl	%esi, (%r8)
.LBB6190:
.LBB6191:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm0
.LBE6191:
.LBE6190:
.LBB6193:
.LBB6187:
	.loc 4 931 0 discriminator 2
	movups	16(%rdx,%rcx), %xmm1
.LBE6187:
.LBE6193:
.LBB6194:
.LBB6192:
	.loc 4 183 0 discriminator 2
	mulps	%xmm2, %xmm0
	addps	%xmm0, %xmm4
.LVL240:
.LBE6192:
.LBE6194:
.LBB6195:
.LBB6196:
	.loc 4 931 0 discriminator 2
	movups	16(%rdx,%rax,4), %xmm0
.LVL241:
.LBE6196:
.LBE6195:
.LBB6197:
.LBB6198:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm0
.LBE6198:
.LBE6197:
.LBB6200:
.LBB6201:
	.loc 4 931 0 discriminator 2
	movups	32(%rdx,%rcx), %xmm1
.LBE6201:
.LBE6200:
.LBB6202:
.LBB6199:
	.loc 4 183 0 discriminator 2
	mulps	%xmm2, %xmm0
	addps	%xmm0, %xmm6
.LVL242:
.LBE6199:
.LBE6202:
.LBB6203:
.LBB6204:
	.loc 4 931 0 discriminator 2
	movups	32(%rdx,%rax,4), %xmm0
.LVL243:
.LBE6204:
.LBE6203:
.LBB6205:
.LBB6206:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm0
.LBE6206:
.LBE6205:
.LBB6208:
.LBB6209:
	.loc 4 931 0 discriminator 2
	movups	48(%rdx,%rax,4), %xmm1
.LBE6209:
.LBE6208:
.LBB6210:
.LBB6207:
	.loc 4 183 0 discriminator 2
	mulps	%xmm2, %xmm0
	addps	%xmm0, %xmm5
.LVL244:
.LBE6207:
.LBE6210:
.LBB6211:
.LBB6212:
	.loc 4 931 0 discriminator 2
	movups	48(%rdx,%rcx), %xmm0
.LVL245:
.LBE6212:
.LBE6211:
.LBB6213:
.LBB6214:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm0
	mulps	%xmm2, %xmm0
	addps	%xmm0, %xmm3
.LVL246:
.LBE6214:
.LBE6213:
	.loc 1 213 0 discriminator 2
	jg	.L108
.LVL247:
.L90:
.LBE6239:
	.loc 1 233 0
	movl	96(%rbx), %ecx
	movl	88(%rbx), %eax
.LBB6240:
.LBB6241:
	.loc 3 815 0
	cvttps2dq	%xmm4, %xmm4
.LVL248:
.LBE6241:
.LBE6240:
.LBB6242:
.LBB6243:
	cvttps2dq	%xmm6, %xmm6
.LVL249:
.LBE6243:
.LBE6242:
.LBB6244:
.LBB6245:
	cvttps2dq	%xmm5, %xmm5
.LVL250:
.LBE6245:
.LBE6244:
.LBB6246:
.LBB6247:
	cvttps2dq	%xmm3, %xmm3
.LVL251:
.LBE6247:
.LBE6246:
.LBB6248:
.LBB6249:
	.loc 3 953 0
	packssdw	%xmm6, %xmm4
.LVL252:
.LBE6249:
.LBE6248:
.LBB6250:
.LBB6251:
	packssdw	%xmm3, %xmm5
.LVL253:
.LBE6251:
.LBE6250:
	.loc 1 233 0
	imull	%ecx, %eax
.LBB6252:
.LBB6253:
	.loc 3 959 0
	packuswb	%xmm5, %xmm4
.LVL254:
.LBE6253:
.LBE6252:
	.loc 1 233 0
	subl	%r10d, %eax
.LVL255:
.LBB6254:
.LBB6255:
	.loc 6 200 0
	cmpl	$15, %eax
.LBE6255:
.LBE6254:
.LBB6257:
.LBB6258:
	.loc 3 710 0
	movaps	%xmm4, 32(%rsp)
.LBE6258:
.LBE6257:
.LBB6260:
.LBB6256:
	.loc 6 200 0
	jg	.L79
.LVL256:
.LBE6256:
.LBE6260:
	.loc 1 235 0
	testl	%eax, %eax
	jg	.L80
.LVL257:
.L89:
.LBE6263:
	.loc 1 200 0
	movl	84(%rbx), %eax
	addl	$16, %r10d
.LVL258:
	addq	$64, %rdx
.LVL259:
	addq	$16, %r11
	movl	%eax, %esi
	imull	%ecx, %esi
	cmpl	%r10d, %esi
	jg	.L110
.LVL260:
.L81:
	addl	$1, %r14d
.LVL261:
	addq	$1, %r15
	cmpl	8(%rsp), %r14d
	jl	.L76
	movl	16(%rsp), %edi
	movl	20(%rsp), %edx
.LVL262:
	movl	24(%rsp), %esi
	addl	%edi, %edx
	movl	%edi, %r14d
.LVL263:
	cmpl	%esi, %edx
	cmovg	%esi, %edx
	movl	%edx, 8(%rsp)
	movl	%edi, %edx
	movl	28(%rsp), %edi
	addl	%edi, %edx
	movl	%edx, 16(%rsp)
	subl	%edi, %edx
	cmpl	%edx, %esi
	jg	.L77
.L74:
	call	GOMP_barrier
.LVL264:
.LBE6274:
.LBE6156:
.LBE6155:
.LBE6154:
	.loc 1 120 0
	movq	56(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L121
	addq	$72, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
.LVL265:
	popq	%rbp
	.cfi_def_cfa_offset 40
.LVL266:
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.LVL267:
	.p2align 4,,10
	.p2align 3
.L79:
	.cfi_restore_state
.LBB6278:
.LBB6277:
.LBB6276:
.LBB6275:
.LBB6264:
.LBB6261:
.LBB6259:
	.loc 3 710 0
	movl	$16, %eax
.LVL268:
.L94:
.LBE6259:
.LBE6261:
.LBB6262:
.LBB6161:
	.loc 5 53 0 discriminator 1
	cmpl	$8, %eax
	movl	%eax, %ecx
	jnb	.L83
	testb	$4, %al
	jne	.L122
	testl	%ecx, %ecx
	je	.L84
	movzbl	(%r12), %eax
	testb	$2, %cl
	movb	%al, (%r11)
	jne	.L123
.L119:
	movl	96(%rbx), %r13d
.L84:
	movl	%r13d, %ecx
	jmp	.L89
	.p2align 4,,10
	.p2align 3
.L83:
	movq	(%r12), %rcx
	movq	%r12, %rdi
	movq	%rcx, (%r11)
	movl	%eax, %ecx
	movq	-8(%r12,%rcx), %rsi
	movq	%rsi, -8(%r11,%rcx)
	leaq	8(%r11), %rsi
	movq	%r11, %rcx
	andq	$-8, %rsi
	subq	%rsi, %rcx
	subq	%rcx, %rdi
	addl	%eax, %ecx
	andl	$-8, %ecx
	cmpl	$8, %ecx
	jb	.L119
	andl	$-8, %ecx
	xorl	%eax, %eax
.L87:
	movl	%eax, %r8d
	addl	$8, %eax
	movq	(%rdi,%r8), %r9
	cmpl	%ecx, %eax
	movq	%r9, (%rsi,%r8)
	jb	.L87
	jmp	.L119
.L122:
	movl	(%r12), %eax
	movl	%eax, (%r11)
	movl	-4(%r12,%rcx), %eax
	movl	%eax, -4(%r11,%rcx)
	movl	96(%rbx), %r13d
	jmp	.L84
.LVL269:
.L80:
	cltq
	jmp	.L94
.LVL270:
.L123:
	movzwl	-2(%r12,%rcx), %eax
	movw	%ax, -2(%r11,%rcx)
	movl	96(%rbx), %r13d
	jmp	.L84
.LVL271:
.L121:
.LBE6161:
.LBE6262:
.LBE6264:
.LBE6275:
.LBE6276:
.LBE6277:
.LBE6278:
	.loc 1 120 0
	call	__stack_chk_fail
.LVL272:
	.cfi_endproc
.LFE12400:
	.size	_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.1, .-_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.1
	.section	.text.unlikely._Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.1,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LCOLDE1:
	.section	.text._Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.1,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LHOTE1:
	.section	.text.unlikely._Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd._omp_fn.2,"axG",@progbits,_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd,comdat
.LCOLDB2:
	.section	.text._Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd._omp_fn.2,"axG",@progbits,_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd,comdat
.LHOTB2:
	.p2align 4,,15
	.type	_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd._omp_fn.2, @function
_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd._omp_fn.2:
.LFB12401:
	.loc 1 120 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
.LVL273:
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rdi, %rbx
	subq	$72, %rsp
	.cfi_def_cfa_offset 128
.LBB6279:
	.loc 1 125 0
	movl	80(%rdi), %r14d
.LBE6279:
	.loc 1 120 0
	movq	40(%rdi), %r12
.LVL274:
	movq	%fs:40, %rax
	movq	%rax, 56(%rsp)
	xorl	%eax, %eax
	movq	32(%rdi), %rbp
.LVL275:
.LBB6410:
	.loc 1 125 0
	leal	3(%r14), %ecx
	testl	%r14d, %r14d
	cmovns	%r14d, %ecx
	sarl	$2, %ecx
	movl	%ecx, 8(%rsp)
	call	omp_get_num_threads
.LVL276:
	movl	%eax, %r15d
	movl	%eax, 20(%rsp)
	call	omp_get_thread_num
.LVL277:
	movl	8(%rsp), %ecx
	movl	%eax, %r10d
	movl	%eax, 16(%rsp)
	imull	%ecx, %r10d
	leal	(%rcx,%r10), %r13d
	cmpl	%r14d, %r13d
	cmovg	%r14d, %r13d
	cmpl	%r10d, %r14d
	jle	.L125
	movl	%r15d, %edi
	movl	%eax, %r11d
	movl	96(%rbx), %edx
	imull	%ecx, %edi
	movl	84(%rbx), %eax
	movl	%edi, 24(%rsp)
	movl	%r15d, %edi
	addl	%r11d, %edi
	movl	%edi, %esi
	movl	%edi, %r15d
	addl	$1, %esi
	imull	%ecx, %r15d
	imull	%esi, %ecx
	subl	%r15d, %ecx
	movl	%ecx, 28(%rsp)
.L128:
	movslq	%r10d, %r11
.L127:
.LVL278:
.LBB6280:
.LBB6281:
	.loc 1 128 0
	movq	48(%rbx), %rdi
.LVL279:
	movq	%r11, %rsi
.LBB6282:
.LBB6283:
	.loc 2 430 0
	movl	76(%rbx), %r8d
.LBE6283:
.LBE6282:
.LBB6286:
	.loc 1 133 0
	xorl	%r9d, %r9d
.LBE6286:
.LBB6405:
.LBB6406:
	.loc 2 430 0
	movq	72(%rdi), %rcx
.LBE6406:
.LBE6405:
.LBB6407:
.LBB6284:
	addl	%r10d, %r8d
	movslq	%r8d, %r8
.LBE6284:
.LBE6407:
	.loc 1 128 0
	imulq	(%rcx), %rsi
	movslq	72(%rbx), %rcx
	addq	%rcx, %rsi
	addq	16(%rdi), %rsi
.LVL280:
	.loc 1 129 0
	movq	56(%rbx), %rdi
.LVL281:
.LBB6408:
.LBB6285:
	.loc 2 430 0
	movq	72(%rdi), %rcx
	imulq	(%rcx), %r8
.LVL282:
.LBE6285:
.LBE6408:
.LBB6409:
	.loc 1 133 0
	movl	%edx, %ecx
	imull	%eax, %ecx
	addq	16(%rdi), %r8
.LVL283:
	testl	%ecx, %ecx
	jle	.L130
.LVL284:
	.p2align 4,,10
	.p2align 3
.L167:
.LBB6287:
.LBB6288:
.LBB6289:
	.loc 3 698 0
	movdqu	(%rsi), %xmm0
.LBE6289:
.LBE6288:
.LBB6290:
.LBB6291:
.LBB6292:
.LBB6293:
	.loc 4 884 0
	movss	(%r12), %xmm6
.LVL285:
	leaq	4(%r12), %rcx
.LBE6293:
.LBE6292:
.LBE6291:
.LBE6290:
.LBB6294:
	.loc 1 149 0
	movq	16(%rbx), %rdi
	movl	$1, %edx
.LBE6294:
	.loc 1 138 0
	movdqa	(%rbx), %xmm5
.LBB6358:
.LBB6359:
	.loc 3 965 0
	movdqa	%xmm0, %xmm7
.LBE6359:
.LBE6358:
.LBB6361:
.LBB6362:
	.loc 4 743 0
	shufps	$0, %xmm6, %xmm6
.LVL286:
.LBE6362:
.LBE6361:
.LBB6363:
.LBB6360:
	.loc 3 965 0
	punpckhbw	%xmm5, %xmm7
.LVL287:
.LBE6360:
.LBE6363:
.LBB6364:
	.loc 1 149 0
	cmpl	$1, (%rdi)
.LBE6364:
.LBB6365:
.LBB6366:
	.loc 3 989 0
	punpcklbw	%xmm5, %xmm0
.LVL288:
.LBE6366:
.LBE6365:
.LBB6367:
.LBB6368:
	.loc 3 995 0
	movdqa	%xmm7, %xmm8
.LBE6368:
.LBE6367:
.LBB6370:
.LBB6371:
	.loc 3 971 0
	movdqa	%xmm0, %xmm9
.LBE6371:
.LBE6370:
.LBB6373:
.LBB6374:
	punpckhwd	%xmm5, %xmm7
.LVL289:
.LBE6374:
.LBE6373:
.LBB6375:
.LBB6376:
	.loc 3 995 0
	punpcklwd	%xmm5, %xmm0
.LVL290:
.LBE6376:
.LBE6375:
.LBB6377:
.LBB6372:
	.loc 3 971 0
	punpckhwd	%xmm5, %xmm9
.LVL291:
.LBE6372:
.LBE6377:
.LBB6378:
.LBB6369:
	.loc 3 995 0
	punpcklwd	%xmm5, %xmm8
.LVL292:
.LBE6369:
.LBE6378:
.LBB6379:
.LBB6380:
	.loc 3 767 0
	cvtdq2ps	%xmm7, %xmm7
.LVL293:
.LBE6380:
.LBE6379:
.LBB6381:
.LBB6382:
	cvtdq2ps	%xmm0, %xmm0
.LVL294:
.LBE6382:
.LBE6381:
.LBB6383:
.LBB6384:
	.loc 4 195 0
	mulps	%xmm6, %xmm7
.LVL295:
.LBE6384:
.LBE6383:
.LBB6385:
.LBB6386:
	.loc 3 767 0
	cvtdq2ps	%xmm9, %xmm9
.LVL296:
.LBE6386:
.LBE6385:
.LBB6387:
.LBB6388:
	cvtdq2ps	%xmm8, %xmm8
.LVL297:
.LBE6388:
.LBE6387:
.LBB6389:
.LBB6390:
	.loc 4 195 0
	mulps	%xmm6, %xmm9
.LBE6390:
.LBE6389:
.LBB6391:
.LBB6392:
	mulps	%xmm6, %xmm8
.LVL298:
.LBE6392:
.LBE6391:
.LBB6393:
.LBB6394:
	mulps	%xmm0, %xmm6
.LVL299:
.LBE6394:
.LBE6393:
.LBB6395:
	.loc 1 149 0
	jle	.L132
	movq	%r8, 8(%rsp)
.LVL300:
	.p2align 4,,10
	.p2align 3
.L168:
.LBB6295:
	.loc 1 152 0 discriminator 2
	movl	96(%rbx), %eax
.LBB6296:
.LBB6297:
	.loc 3 698 0 discriminator 2
	movq	%rsi, %r8
.LBE6297:
.LBE6296:
.LBB6300:
.LBB6301:
.LBB6302:
.LBB6303:
	.loc 4 884 0 discriminator 2
	movss	(%rcx), %xmm0
.LVL301:
	addq	$4, %rcx
.LBE6303:
.LBE6302:
.LBE6301:
.LBE6300:
.LBB6304:
.LBB6305:
	.loc 4 743 0 discriminator 2
	shufps	$0, %xmm0, %xmm0
.LVL302:
.LBE6305:
.LBE6304:
	.loc 1 152 0 discriminator 2
	imull	%edx, %eax
.LBE6295:
	.loc 1 149 0 discriminator 2
	addl	$1, %edx
.LVL303:
.LBB6356:
	.loc 1 152 0 discriminator 2
	cltq
.LVL304:
.LBB6306:
.LBB6298:
	.loc 3 698 0 discriminator 2
	subq	%rax, %r8
.LVL305:
.LBE6298:
.LBE6306:
.LBB6307:
.LBB6308:
	movdqu	(%rsi,%rax), %xmm3
.LVL306:
.LBE6308:
.LBE6307:
.LBE6356:
	.loc 1 149 0 discriminator 2
	cmpl	%edx, (%rdi)
.LBB6357:
.LBB6309:
.LBB6299:
	.loc 3 698 0 discriminator 2
	movdqu	(%r8), %xmm4
.LVL307:
.LBE6299:
.LBE6309:
.LBB6310:
.LBB6311:
	.loc 3 965 0 discriminator 2
	movdqa	%xmm3, %xmm1
.LBE6311:
.LBE6310:
.LBB6313:
.LBB6314:
	.loc 3 989 0 discriminator 2
	punpcklbw	%xmm5, %xmm3
.LVL308:
.LBE6314:
.LBE6313:
.LBB6315:
.LBB6316:
	.loc 3 965 0 discriminator 2
	movdqa	%xmm4, %xmm2
.LBE6316:
.LBE6315:
.LBB6318:
.LBB6312:
	punpckhbw	%xmm5, %xmm1
.LVL309:
.LBE6312:
.LBE6318:
.LBB6319:
.LBB6320:
	.loc 3 989 0 discriminator 2
	punpcklbw	%xmm5, %xmm4
.LVL310:
.LBE6320:
.LBE6319:
.LBB6321:
.LBB6317:
	.loc 3 965 0 discriminator 2
	punpckhbw	%xmm5, %xmm2
.LVL311:
.LBE6317:
.LBE6321:
.LBB6322:
.LBB6323:
	.loc 3 971 0 discriminator 2
	paddw	%xmm4, %xmm3
	movdqa	%xmm3, %xmm4
.LVL312:
.LBE6323:
.LBE6322:
.LBB6325:
.LBB6326:
	.loc 3 995 0 discriminator 2
	paddw	%xmm2, %xmm1
	movdqa	%xmm1, %xmm2
.LVL313:
.LBE6326:
.LBE6325:
.LBB6328:
.LBB6329:
	.loc 3 971 0 discriminator 2
	punpckhwd	%xmm5, %xmm1
.LBE6329:
.LBE6328:
.LBB6330:
.LBB6324:
	punpckhwd	%xmm5, %xmm4
.LVL314:
.LBE6324:
.LBE6330:
.LBB6331:
.LBB6327:
	.loc 3 995 0 discriminator 2
	punpcklwd	%xmm5, %xmm2
.LBE6327:
.LBE6331:
.LBB6332:
.LBB6333:
	.loc 3 767 0 discriminator 2
	cvtdq2ps	%xmm1, %xmm1
.LBE6333:
.LBE6332:
.LBB6334:
.LBB6335:
	.loc 4 183 0 discriminator 2
	mulps	%xmm0, %xmm1
.LBE6335:
.LBE6334:
.LBB6337:
.LBB6338:
	.loc 3 995 0 discriminator 2
	punpcklwd	%xmm5, %xmm3
.LVL315:
.LBE6338:
.LBE6337:
.LBB6339:
.LBB6336:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm7
.LVL316:
.LBE6336:
.LBE6339:
.LBB6340:
.LBB6341:
	.loc 3 767 0 discriminator 2
	cvtdq2ps	%xmm2, %xmm1
.LVL317:
.LBE6341:
.LBE6340:
.LBB6342:
.LBB6343:
	cvtdq2ps	%xmm3, %xmm3
.LVL318:
.LBE6343:
.LBE6342:
.LBB6344:
.LBB6345:
	.loc 4 183 0 discriminator 2
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm8
.LVL319:
.LBE6345:
.LBE6344:
.LBB6346:
.LBB6347:
	.loc 3 767 0 discriminator 2
	cvtdq2ps	%xmm4, %xmm1
.LVL320:
.LBE6347:
.LBE6346:
.LBB6348:
.LBB6349:
	.loc 4 183 0 discriminator 2
	mulps	%xmm0, %xmm1
.LBE6349:
.LBE6348:
.LBB6351:
.LBB6352:
	mulps	%xmm3, %xmm0
.LVL321:
.LBE6352:
.LBE6351:
.LBB6354:
.LBB6350:
	addps	%xmm1, %xmm9
.LVL322:
.LBE6350:
.LBE6354:
.LBB6355:
.LBB6353:
	addps	%xmm0, %xmm6
.LVL323:
.LBE6353:
.LBE6355:
.LBE6357:
	.loc 1 149 0 discriminator 2
	jg	.L168
	movq	8(%rsp), %r8
.LVL324:
.L132:
.LBE6395:
.LBB6396:
.LBB6397:
	.loc 4 980 0
	movups	%xmm6, (%r8)
.LVL325:
.LBE6397:
.LBE6396:
.LBE6287:
	.loc 1 133 0
	addl	$16, %r9d
.LVL326:
	addq	$16, %rsi
.LVL327:
	addq	$64, %r8
.LVL328:
.LBB6404:
.LBB6398:
.LBB6399:
	.loc 4 980 0
	movups	%xmm9, -48(%r8)
.LVL329:
.LBE6399:
.LBE6398:
.LBB6400:
.LBB6401:
	movups	%xmm8, -32(%r8)
.LVL330:
.LBE6401:
.LBE6400:
.LBB6402:
.LBB6403:
	movups	%xmm7, -16(%r8)
.LVL331:
.LBE6403:
.LBE6402:
.LBE6404:
	.loc 1 133 0
	movl	96(%rbx), %edx
	movl	84(%rbx), %eax
	movl	%edx, %ecx
	imull	%eax, %ecx
	cmpl	%r9d, %ecx
	jg	.L167
.LVL332:
.L130:
	addl	$1, %r10d
.LVL333:
	addq	$1, %r11
	cmpl	%r13d, %r10d
	jl	.L127
	movl	28(%rsp), %edi
	movl	%r15d, %r10d
.LVL334:
	leal	(%r15,%rdi), %r13d
	movl	24(%rsp), %edi
	cmpl	%r14d, %r13d
	cmovg	%r14d, %r13d
	addl	%edi, %r15d
	movl	%r15d, %ecx
	subl	%edi, %ecx
	cmpl	%ecx, %r14d
	jg	.L128
.LVL335:
.L125:
	call	GOMP_barrier
.LVL336:
.LBE6409:
.LBE6281:
.LBE6280:
.LBE6410:
.LBB6411:
	.loc 1 179 0
	movl	16(%rsp), %eax
	testl	%eax, %eax
	jne	.L153
.LBB6412:
	.loc 1 180 0
	movq	56(%rbx), %rdx
	.loc 1 181 0
	movslq	76(%rbx), %r14
	movl	92(%rbx), %eax
	movq	16(%rdx), %r15
.LVL337:
.LBB6413:
.LBB6414:
	.loc 2 430 0
	movq	72(%rdx), %rdx
.LVL338:
.LBE6414:
.LBE6413:
	.loc 1 181 0
	subl	%r14d, %eax
.LVL339:
	movq	%r14, %rcx
.LBB6419:
.LBB6415:
	.loc 2 430 0
	movslq	%eax, %r12
.LVL340:
.LBE6415:
.LBE6419:
.LBB6420:
.LBB6421:
	subl	$1, %eax
.LVL341:
.LBE6421:
.LBE6420:
.LBB6426:
.LBB6416:
	movq	(%rdx), %rdx
.LBE6416:
.LBE6426:
.LBB6427:
.LBB6422:
	cltq
.LBE6422:
.LBE6427:
.LBB6428:
.LBB6417:
	imulq	%rdx, %r12
.LBE6417:
.LBE6428:
.LBB6429:
.LBB6430:
	imulq	%rdx, %r14
.LBE6430:
.LBE6429:
.LBB6432:
.LBB6423:
	imulq	%rdx, %rax
.LVL342:
.LBE6423:
.LBE6432:
.LBB6433:
.LBB6418:
	addq	%r15, %r12
.LVL343:
.LBE6418:
.LBE6433:
.LBB6434:
.LBB6431:
	addq	%r15, %r14
.LVL344:
.LBE6431:
.LBE6434:
.LBB6435:
.LBB6424:
	addq	%r15, %rax
.LBE6424:
.LBE6435:
.LBB6436:
	.loc 1 184 0
	testl	%ecx, %ecx
.LBE6436:
.LBB6443:
.LBB6425:
	.loc 2 430 0
	movq	%rax, 8(%rsp)
.LVL345:
.LBE6425:
.LBE6443:
.LBB6444:
	.loc 1 184 0
	jle	.L153
	movslq	84(%rbx), %rax
	xorl	%r13d, %r13d
.LVL346:
.L154:
.LBB6437:
.LBB6438:
	.loc 5 53 0 discriminator 2
	leaq	0(,%rax,4), %rdx
	movq	%r15, %rdi
	movq	%r14, %rsi
.LBE6438:
.LBE6437:
	.loc 1 184 0 discriminator 2
	addl	$1, %r13d
.LVL347:
.LBB6440:
.LBB6439:
	.loc 5 53 0 discriminator 2
	call	memcpy
.LVL348:
.LBE6439:
.LBE6440:
.LBB6441:
.LBB6442:
	movslq	84(%rbx), %rdx
	movq	8(%rsp), %rsi
	movq	%r12, %rdi
	salq	$2, %rdx
.LVL349:
	call	memcpy
.LVL350:
.LBE6442:
.LBE6441:
	.loc 1 184 0 discriminator 2
	movslq	84(%rbx), %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %r15
.LVL351:
	addq	%rdx, %r12
.LVL352:
	cmpl	%r13d, 76(%rbx)
	jg	.L154
.LVL353:
.L153:
.LBE6444:
.LBE6412:
.LBE6411:
.LBB6445:
	.loc 1 192 0
	movl	80(%rbx), %eax
	movl	16(%rsp), %r15d
	leal	3(%rax), %edx
	testl	%eax, %eax
	movl	%r15d, %r14d
	movl	%eax, %edi
	movl	%eax, 24(%rsp)
	cmovns	%eax, %edx
	sarl	$2, %edx
	imull	%edx, %r14d
	leal	(%rdx,%r14), %eax
	cmpl	%edi, %eax
	movl	%eax, %esi
	cmovg	%edi, %esi
	cmpl	%r14d, %edi
	movl	%esi, 8(%rsp)
	jle	.L135
	movl	20(%rsp), %esi
	movl	96(%rbx), %r13d
.LBB6446:
.LBB6447:
.LBB6448:
.LBB6449:
.LBB6450:
.LBB6451:
	.loc 5 53 0
	leaq	32(%rsp), %r12
	movl	84(%rbx), %eax
	movl	%esi, %edi
	movl	%r13d, %ecx
	imull	%edx, %edi
	movl	%edi, 28(%rsp)
	movl	%esi, %edi
	addl	%r15d, %edi
	movl	%edi, %esi
	addl	$1, %esi
	imull	%edx, %edi
	imull	%esi, %edx
	movl	%edi, 16(%rsp)
	subl	%edi, %edx
	movl	%edx, 20(%rsp)
.L138:
	movslq	%r14d, %r15
.L137:
.LVL354:
.LBE6451:
.LBE6450:
.LBE6449:
.LBE6448:
	.loc 1 195 0
	movq	56(%rbx), %rsi
.LVL355:
.LBB6556:
.LBB6557:
	.loc 2 430 0
	movl	76(%rbx), %edx
.LBE6557:
.LBE6556:
.LBB6559:
.LBB6560:
	movq	%r15, %r11
.LBE6560:
.LBE6559:
.LBB6562:
	.loc 1 200 0
	xorl	%r10d, %r10d
.LBE6562:
.LBB6563:
.LBB6558:
	.loc 2 430 0
	movq	72(%rsi), %rdi
	addl	%r14d, %edx
.LVL356:
	movslq	%edx, %rdx
	imulq	(%rdi), %rdx
.LVL357:
	addq	16(%rsi), %rdx
.LVL358:
.LBE6558:
.LBE6563:
	.loc 1 196 0
	movq	64(%rbx), %rsi
.LVL359:
.LBB6564:
.LBB6561:
	.loc 2 430 0
	movq	72(%rsi), %rdi
	imulq	(%rdi), %r11
.LVL360:
.LBE6561:
.LBE6564:
.LBB6565:
	.loc 1 200 0
	movl	%ecx, %edi
	imull	%eax, %edi
	addq	16(%rsi), %r11
.LVL361:
	testl	%edi, %edi
	jle	.L142
.LVL362:
	.p2align 4,,10
	.p2align 3
.L171:
.LBB6554:
.LBB6453:
	.loc 1 213 0
	movq	24(%rbx), %r8
.LBE6453:
.LBB6506:
.LBB6507:
.LBB6508:
.LBB6509:
	.loc 4 884 0
	movss	0(%rbp), %xmm3
.LVL363:
	leaq	4(%rbp), %rdi
.LBE6509:
.LBE6508:
.LBE6507:
.LBE6506:
.LBB6510:
.LBB6511:
	.loc 4 931 0
	movups	(%rdx), %xmm4
.LBE6511:
.LBE6510:
.LBB6512:
	.loc 1 213 0
	movl	$1, %esi
.LBE6512:
.LBB6513:
.LBB6514:
	.loc 4 743 0
	shufps	$0, %xmm3, %xmm3
.LVL364:
.LBE6514:
.LBE6513:
.LBB6515:
	.loc 1 213 0
	cmpl	$1, (%r8)
.LBE6515:
.LBB6516:
.LBB6517:
	.loc 4 931 0
	movups	16(%rdx), %xmm6
.LVL365:
.LBE6517:
.LBE6516:
.LBB6518:
.LBB6519:
	.loc 4 195 0
	mulps	%xmm3, %xmm4
.LVL366:
.LBE6519:
.LBE6518:
.LBB6520:
.LBB6521:
	.loc 4 931 0
	movups	32(%rdx), %xmm5
.LVL367:
.LBE6521:
.LBE6520:
.LBB6522:
.LBB6523:
	.loc 4 195 0
	mulps	%xmm3, %xmm6
.LVL368:
.LBE6523:
.LBE6522:
.LBB6524:
.LBB6525:
	.loc 4 931 0
	movups	48(%rdx), %xmm0
.LVL369:
.LBE6525:
.LBE6524:
.LBB6526:
.LBB6527:
	.loc 4 195 0
	mulps	%xmm3, %xmm5
.LVL370:
.LBE6527:
.LBE6526:
.LBB6528:
.LBB6529:
	mulps	%xmm0, %xmm3
.LVL371:
.LBE6529:
.LBE6528:
.LBB6530:
	.loc 1 213 0
	jle	.L151
.LVL372:
	.p2align 4,,10
	.p2align 3
.L169:
.LBB6454:
.LBB6455:
	.loc 4 931 0 discriminator 2
	movl	84(%rbx), %eax
.LBE6455:
.LBE6454:
.LBB6461:
.LBB6462:
.LBB6463:
.LBB6464:
	.loc 4 884 0 discriminator 2
	movss	(%rdi), %xmm0
.LVL373:
.LBE6464:
.LBE6463:
.LBE6462:
.LBE6461:
.LBB6465:
.LBB6456:
	.loc 4 931 0 discriminator 2
	movq	%rdx, %r9
	addq	$4, %rdi
.LBE6456:
.LBE6465:
.LBB6466:
.LBB6467:
	.loc 4 743 0 discriminator 2
	movaps	%xmm0, %xmm2
.LBE6467:
.LBE6466:
.LBB6469:
.LBB6457:
	.loc 4 931 0 discriminator 2
	imull	%esi, %eax
.LBE6457:
.LBE6469:
	.loc 1 213 0 discriminator 2
	addl	$1, %esi
.LVL374:
.LBB6470:
.LBB6458:
	.loc 4 931 0 discriminator 2
	imull	96(%rbx), %eax
.LBE6458:
.LBE6470:
.LBB6471:
.LBB6468:
	.loc 4 743 0 discriminator 2
	shufps	$0, %xmm0, %xmm2
.LVL375:
.LBE6468:
.LBE6471:
.LBB6472:
.LBB6459:
	.loc 4 931 0 discriminator 2
	cltq
	leaq	0(,%rax,4), %rcx
	subq	%rcx, %r9
.LBE6459:
.LBE6472:
.LBB6473:
.LBB6474:
	movups	(%rdx,%rcx), %xmm0
.LBE6474:
.LBE6473:
.LBB6475:
.LBB6476:
	movq	%rax, %rcx
.LBE6476:
.LBE6475:
.LBB6479:
.LBB6460:
	movups	(%r9), %xmm1
.LVL376:
.LBE6460:
.LBE6479:
.LBB6480:
.LBB6477:
	negq	%rcx
	salq	$2, %rcx
.LBE6477:
.LBE6480:
	.loc 1 213 0 discriminator 2
	cmpl	%esi, (%r8)
.LBB6481:
.LBB6482:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm0
.LBE6482:
.LBE6481:
.LBB6484:
.LBB6478:
	.loc 4 931 0 discriminator 2
	movups	16(%rdx,%rcx), %xmm1
.LBE6478:
.LBE6484:
.LBB6485:
.LBB6483:
	.loc 4 183 0 discriminator 2
	mulps	%xmm2, %xmm0
	addps	%xmm0, %xmm4
.LVL377:
.LBE6483:
.LBE6485:
.LBB6486:
.LBB6487:
	.loc 4 931 0 discriminator 2
	movups	16(%rdx,%rax,4), %xmm0
.LVL378:
.LBE6487:
.LBE6486:
.LBB6488:
.LBB6489:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm0
.LBE6489:
.LBE6488:
.LBB6491:
.LBB6492:
	.loc 4 931 0 discriminator 2
	movups	32(%rdx,%rcx), %xmm1
.LBE6492:
.LBE6491:
.LBB6493:
.LBB6490:
	.loc 4 183 0 discriminator 2
	mulps	%xmm2, %xmm0
	addps	%xmm0, %xmm6
.LVL379:
.LBE6490:
.LBE6493:
.LBB6494:
.LBB6495:
	.loc 4 931 0 discriminator 2
	movups	32(%rdx,%rax,4), %xmm0
.LVL380:
.LBE6495:
.LBE6494:
.LBB6496:
.LBB6497:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm0
.LBE6497:
.LBE6496:
.LBB6499:
.LBB6500:
	.loc 4 931 0 discriminator 2
	movups	48(%rdx,%rax,4), %xmm1
.LBE6500:
.LBE6499:
.LBB6501:
.LBB6498:
	.loc 4 183 0 discriminator 2
	mulps	%xmm2, %xmm0
	addps	%xmm0, %xmm5
.LVL381:
.LBE6498:
.LBE6501:
.LBB6502:
.LBB6503:
	.loc 4 931 0 discriminator 2
	movups	48(%rdx,%rcx), %xmm0
.LVL382:
.LBE6503:
.LBE6502:
.LBB6504:
.LBB6505:
	.loc 4 183 0 discriminator 2
	addps	%xmm1, %xmm0
	mulps	%xmm2, %xmm0
	addps	%xmm0, %xmm3
.LVL383:
.LBE6505:
.LBE6504:
	.loc 1 213 0 discriminator 2
	jg	.L169
.LVL384:
.L151:
.LBE6530:
	.loc 1 233 0
	movl	96(%rbx), %ecx
	movl	88(%rbx), %eax
.LBB6531:
.LBB6532:
	.loc 3 815 0
	cvttps2dq	%xmm4, %xmm4
.LVL385:
.LBE6532:
.LBE6531:
.LBB6533:
.LBB6534:
	cvttps2dq	%xmm6, %xmm6
.LVL386:
.LBE6534:
.LBE6533:
.LBB6535:
.LBB6536:
	cvttps2dq	%xmm5, %xmm5
.LVL387:
.LBE6536:
.LBE6535:
.LBB6537:
.LBB6538:
	cvttps2dq	%xmm3, %xmm3
.LVL388:
.LBE6538:
.LBE6537:
.LBB6539:
.LBB6540:
	.loc 3 953 0
	packssdw	%xmm6, %xmm4
.LVL389:
.LBE6540:
.LBE6539:
.LBB6541:
.LBB6542:
	packssdw	%xmm3, %xmm5
.LVL390:
.LBE6542:
.LBE6541:
	.loc 1 233 0
	imull	%ecx, %eax
.LBB6543:
.LBB6544:
	.loc 3 959 0
	packuswb	%xmm5, %xmm4
.LVL391:
.LBE6544:
.LBE6543:
	.loc 1 233 0
	subl	%r10d, %eax
.LVL392:
.LBB6545:
.LBB6546:
	.loc 6 200 0
	cmpl	$15, %eax
.LBE6546:
.LBE6545:
.LBB6548:
.LBB6549:
	.loc 3 710 0
	movaps	%xmm4, 32(%rsp)
.LBE6549:
.LBE6548:
.LBB6551:
.LBB6547:
	.loc 6 200 0
	jg	.L140
.LVL393:
.LBE6547:
.LBE6551:
	.loc 1 235 0
	testl	%eax, %eax
	jg	.L141
.LVL394:
.L150:
.LBE6554:
	.loc 1 200 0
	movl	84(%rbx), %eax
	addl	$16, %r10d
.LVL395:
	addq	$64, %rdx
.LVL396:
	addq	$16, %r11
	movl	%eax, %esi
	imull	%ecx, %esi
	cmpl	%r10d, %esi
	jg	.L171
.LVL397:
.L142:
	addl	$1, %r14d
.LVL398:
	addq	$1, %r15
	cmpl	8(%rsp), %r14d
	jl	.L137
	movl	16(%rsp), %edi
	movl	20(%rsp), %edx
.LVL399:
	movl	24(%rsp), %esi
	addl	%edi, %edx
	movl	%edi, %r14d
.LVL400:
	cmpl	%esi, %edx
	cmovg	%esi, %edx
	movl	%edx, 8(%rsp)
	movl	%edi, %edx
	movl	28(%rsp), %edi
	addl	%edi, %edx
	movl	%edx, 16(%rsp)
	subl	%edi, %edx
	cmpl	%edx, %esi
	jg	.L138
.L135:
	call	GOMP_barrier
.LVL401:
.LBE6565:
.LBE6447:
.LBE6446:
.LBE6445:
	.loc 1 120 0
	movq	56(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L182
	addq	$72, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
.LVL402:
	popq	%rbp
	.cfi_def_cfa_offset 40
.LVL403:
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.LVL404:
	.p2align 4,,10
	.p2align 3
.L140:
	.cfi_restore_state
.LBB6569:
.LBB6568:
.LBB6567:
.LBB6566:
.LBB6555:
.LBB6552:
.LBB6550:
	.loc 3 710 0
	movl	$16, %eax
.LVL405:
.L155:
.LBE6550:
.LBE6552:
.LBB6553:
.LBB6452:
	.loc 5 53 0 discriminator 1
	cmpl	$8, %eax
	movl	%eax, %ecx
	jnb	.L144
	testb	$4, %al
	jne	.L183
	testl	%ecx, %ecx
	je	.L145
	movzbl	(%r12), %eax
	testb	$2, %cl
	movb	%al, (%r11)
	jne	.L184
.L180:
	movl	96(%rbx), %r13d
.L145:
	movl	%r13d, %ecx
	jmp	.L150
	.p2align 4,,10
	.p2align 3
.L144:
	movq	(%r12), %rcx
	movq	%r12, %rdi
	movq	%rcx, (%r11)
	movl	%eax, %ecx
	movq	-8(%r12,%rcx), %rsi
	movq	%rsi, -8(%r11,%rcx)
	leaq	8(%r11), %rsi
	movq	%r11, %rcx
	andq	$-8, %rsi
	subq	%rsi, %rcx
	subq	%rcx, %rdi
	addl	%eax, %ecx
	andl	$-8, %ecx
	cmpl	$8, %ecx
	jb	.L180
	andl	$-8, %ecx
	xorl	%eax, %eax
.L148:
	movl	%eax, %r8d
	addl	$8, %eax
	movq	(%rdi,%r8), %r9
	cmpl	%ecx, %eax
	movq	%r9, (%rsi,%r8)
	jb	.L148
	jmp	.L180
.L183:
	movl	(%r12), %eax
	movl	%eax, (%r11)
	movl	-4(%r12,%rcx), %eax
	movl	%eax, -4(%r11,%rcx)
	movl	96(%rbx), %r13d
	jmp	.L145
.LVL406:
.L141:
	cltq
	jmp	.L155
.LVL407:
.L184:
	movzwl	-2(%r12,%rcx), %eax
	movw	%ax, -2(%r11,%rcx)
	movl	96(%rbx), %r13d
	jmp	.L145
.LVL408:
.L182:
.LBE6452:
.LBE6553:
.LBE6555:
.LBE6566:
.LBE6567:
.LBE6568:
.LBE6569:
	.loc 1 120 0
	call	__stack_chk_fail
.LVL409:
	.cfi_endproc
.LFE12401:
	.size	_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd._omp_fn.2, .-_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd._omp_fn.2
	.section	.text.unlikely._Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd._omp_fn.2,"axG",@progbits,_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd,comdat
.LCOLDE2:
	.section	.text._Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd._omp_fn.2,"axG",@progbits,_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd,comdat
.LHOTE2:
	.section	.text.unlikely._ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_._omp_fn.3,"axG",@progbits,_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_,comdat
.LCOLDB3:
	.section	.text._ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_._omp_fn.3,"axG",@progbits,_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_,comdat
.LHOTB3:
	.p2align 4,,15
	.type	_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_._omp_fn.3, @function
_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_._omp_fn.3:
.LFB12402:
	.loc 1 392 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
.LVL410:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movl	$1, %ecx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	leaq	-64(%rbp), %r9
	pushq	%rbx
	leaq	-72(%rbp), %r8
	subq	$216, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movslq	76(%rdi), %rsi
	.loc 1 392 0
	movq	16(%rdi), %r14
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
	movq	24(%rdi), %rax
	movq	%rdi, -88(%rbp)
	movq	%rax, -232(%rbp)
.LVL411:
	movslq	80(%rdi), %rax
.LVL412:
	xorl	%edi, %edi
.LVL413:
	movq	%rax, %rdx
	movl	%eax, -96(%rbp)
.LVL414:
	movq	%rax, %rbx
	movq	%rax, -248(%rbp)
	call	GOMP_loop_dynamic_start
.LVL415:
	testb	%al, %al
	movq	-88(%rbp), %r10
	je	.L186
	movq	%rbx, %rax
	movq	%r14, %r13
	salq	$2, %rax
	movq	%rax, -240(%rbp)
	leaq	96(%r14), %rax
	movq	%r10, %r14
.LVL416:
	movq	%rax, -224(%rbp)
.LVL417:
.L190:
	movq	-72(%rbp), %rax
	movl	-64(%rbp), %ebx
	movq	%r14, %r15
	movq	%r13, %r14
	movl	%eax, -172(%rbp)
.LVL418:
	movl	%eax, -176(%rbp)
	cltq
.LVL419:
	movq	%rax, -192(%rbp)
	salq	$2, %rax
	movl	%ebx, -216(%rbp)
	movq	%rax, -184(%rbp)
.L189:
.LVL420:
.LBB6570:
.LBB6571:
.LBB6572:
	.loc 1 399 0
	movl	72(%r15), %r9d
	movq	-232(%rbp), %rax
	xorl	%esi, %esi
	addq	-184(%rbp), %rax
	movq	$0, -104(%rbp)
	testl	%r9d, %r9d
	movq	%rax, -120(%rbp)
	jle	.L213
.LVL421:
	.p2align 4,,10
	.p2align 3
.L263:
.LBB6573:
	.loc 1 400 0
	movq	32(%r15), %rax
	movq	-104(%rbp), %rbx
	movq	16(%rax), %rdi
.LVL422:
.LBB6574:
.LBB6575:
	.loc 2 436 0
	movq	72(%rax), %rax
.LVL423:
.LBE6575:
.LBE6574:
	.loc 1 400 0
	movq	%rbx, %r8
	imulq	(%rax), %r8
	movq	-192(%rbp), %rax
	addq	%rax, %r8
	addq	%rdi, %r8
.LVL424:
	.loc 1 401 0
	addq	%rax, %rdi
.LVL425:
	.loc 1 402 0
	movq	48(%r15), %rax
.LVL426:
.LBB6576:
.LBB6577:
	.loc 2 430 0
	movq	72(%rax), %rdx
.LBE6577:
.LBE6576:
	.loc 1 402 0
	imulq	(%rdx), %rbx
	movq	%rbx, %rdx
	addq	-184(%rbp), %rdx
	addq	16(%rax), %rdx
.LVL427:
.LBB6578:
	.loc 1 405 0
	movl	-96(%rbp), %eax
.LVL428:
	testl	%eax, %eax
	jle	.L210
	movl	76(%r15), %eax
	leal	-15(%rax), %ecx
	cmpl	%ecx, -172(%rbp)
	jge	.L282
	leaq	80(%r14), %r13
	leaq	64(%r14), %r12
	movq	%r14, -112(%rbp)
	movq	-120(%rbp), %rbx
	movl	-176(%rbp), %r14d
	xorl	%r9d, %r9d
.LVL429:
	.p2align 4,,10
	.p2align 3
.L228:
	pxor	%xmm1, %xmm1
	movl	$1, %ecx
	movdqa	(%r15), %xmm5
	xorl	%eax, %eax
	movl	%r9d, -88(%rbp)
	movaps	%xmm1, %xmm7
	movaps	%xmm1, %xmm6
	movaps	%xmm1, %xmm8
.LVL430:
.L233:
.LBB6579:
.LBB6580:
.LBB6581:
	.loc 1 409 0
	cmpl	%eax, %esi
	movl	%eax, %r11d
.LVL431:
	jge	.L230
	movq	%rdi, %r10
.L235:
.LVL432:
.LBB6582:
.LBB6583:
	.loc 3 698 0 discriminator 4
	movdqu	(%r10), %xmm2
.LBE6583:
.LBE6582:
.LBB6584:
.LBB6585:
.LBB6586:
.LBB6587:
	.loc 4 884 0 discriminator 4
	movss	0(%r13,%rax,4), %xmm0
.LVL433:
.LBE6587:
.LBE6586:
.LBE6585:
.LBE6584:
	.loc 1 424 0 discriminator 4
	cmpl	%r11d, %esi
.LBB6588:
.LBB6589:
	.loc 3 965 0 discriminator 4
	movdqa	%xmm2, %xmm3
.LBE6589:
.LBE6588:
.LBB6591:
.LBB6592:
	.loc 4 743 0 discriminator 4
	shufps	$0, %xmm0, %xmm0
.LVL434:
.LBE6592:
.LBE6591:
.LBB6593:
.LBB6590:
	.loc 3 965 0 discriminator 4
	punpckhbw	%xmm5, %xmm3
.LVL435:
.LBE6590:
.LBE6593:
.LBB6594:
.LBB6595:
	.loc 3 989 0 discriminator 4
	punpcklbw	%xmm5, %xmm2
.LVL436:
.LBE6595:
.LBE6594:
.LBB6596:
.LBB6597:
	.loc 3 995 0 discriminator 4
	movdqa	%xmm3, %xmm9
.LBE6597:
.LBE6596:
.LBB6599:
.LBB6600:
	.loc 3 971 0 discriminator 4
	punpckhwd	%xmm5, %xmm3
.LVL437:
.LBE6600:
.LBE6599:
.LBB6601:
.LBB6602:
	movdqa	%xmm2, %xmm4
.LBE6602:
.LBE6601:
.LBB6604:
.LBB6598:
	.loc 3 995 0 discriminator 4
	punpcklwd	%xmm5, %xmm9
.LBE6598:
.LBE6604:
.LBB6605:
.LBB6603:
	.loc 3 971 0 discriminator 4
	punpckhwd	%xmm5, %xmm4
.LVL438:
.LBE6603:
.LBE6605:
.LBB6606:
.LBB6607:
	.loc 3 767 0 discriminator 4
	cvtdq2ps	%xmm3, %xmm3
.LBE6607:
.LBE6606:
.LBB6608:
.LBB6609:
	.loc 4 183 0 discriminator 4
	mulps	%xmm0, %xmm3
.LBE6609:
.LBE6608:
.LBB6611:
.LBB6612:
	.loc 3 995 0 discriminator 4
	punpcklwd	%xmm5, %xmm2
.LVL439:
.LBE6612:
.LBE6611:
.LBB6613:
.LBB6614:
	.loc 3 767 0 discriminator 4
	cvtdq2ps	%xmm4, %xmm4
.LVL440:
.LBE6614:
.LBE6613:
.LBB6615:
.LBB6616:
	.loc 4 183 0 discriminator 4
	mulps	%xmm0, %xmm4
.LBE6616:
.LBE6615:
.LBB6618:
.LBB6610:
	addps	%xmm3, %xmm8
.LVL441:
.LBE6610:
.LBE6618:
.LBB6619:
.LBB6620:
	.loc 3 767 0 discriminator 4
	cvtdq2ps	%xmm2, %xmm2
.LVL442:
.LBE6620:
.LBE6619:
.LBB6621:
.LBB6622:
	cvtdq2ps	%xmm9, %xmm3
.LVL443:
.LBE6622:
.LBE6621:
.LBB6623:
.LBB6624:
	.loc 4 183 0 discriminator 4
	mulps	%xmm0, %xmm3
.LBE6624:
.LBE6623:
.LBB6626:
.LBB6627:
	mulps	%xmm2, %xmm0
.LVL444:
.LBE6627:
.LBE6626:
.LBB6629:
.LBB6617:
	addps	%xmm4, %xmm7
.LVL445:
.LBE6617:
.LBE6629:
.LBB6630:
.LBB6625:
	addps	%xmm3, %xmm6
.LVL446:
.LBE6625:
.LBE6630:
.LBB6631:
.LBB6628:
	addps	%xmm0, %xmm1
.LVL447:
.LBE6628:
.LBE6631:
	.loc 1 424 0 discriminator 4
	jg	.L231
	movq	%rbx, %r10
.LVL448:
.L234:
.LBB6632:
.LBB6633:
.LBB6634:
.LBB6635:
	.loc 4 884 0 discriminator 4
	movss	(%r12,%rax,4), %xmm0
.LVL449:
.LBE6635:
.LBE6634:
.LBE6633:
.LBE6632:
.LBB6636:
.LBB6637:
	.loc 4 931 0 discriminator 4
	movups	48(%r10), %xmm9
.LVL450:
	addq	$1, %rax
.LVL451:
	addl	$1, %ecx
.LBE6637:
.LBE6636:
.LBB6638:
.LBB6639:
	.loc 4 743 0 discriminator 4
	shufps	$0, %xmm0, %xmm0
.LVL452:
.LBE6639:
.LBE6638:
.LBB6640:
.LBB6641:
	.loc 4 931 0 discriminator 4
	movups	32(%r10), %xmm4
.LVL453:
.LBE6641:
.LBE6640:
.LBE6581:
	.loc 1 408 0 discriminator 4
	cmpq	$4, %rax
.LBB6662:
.LBB6642:
.LBB6643:
	.loc 4 931 0 discriminator 4
	movups	16(%r10), %xmm3
.LVL454:
.LBE6643:
.LBE6642:
.LBB6644:
.LBB6645:
	.loc 4 189 0 discriminator 4
	mulps	%xmm0, %xmm9
.LVL455:
.LBE6645:
.LBE6644:
.LBB6647:
.LBB6648:
	.loc 4 931 0 discriminator 4
	movups	(%r10), %xmm2
.LVL456:
.LBE6648:
.LBE6647:
.LBB6649:
.LBB6650:
	.loc 4 189 0 discriminator 4
	mulps	%xmm0, %xmm4
.LVL457:
.LBE6650:
.LBE6649:
.LBB6652:
.LBB6653:
	mulps	%xmm0, %xmm3
.LVL458:
.LBE6653:
.LBE6652:
.LBB6655:
.LBB6646:
	subps	%xmm9, %xmm8
.LVL459:
.LBE6646:
.LBE6655:
.LBB6656:
.LBB6657:
	mulps	%xmm2, %xmm0
.LVL460:
.LBE6657:
.LBE6656:
.LBB6659:
.LBB6651:
	subps	%xmm4, %xmm6
.LVL461:
.LBE6651:
.LBE6659:
.LBB6660:
.LBB6654:
	subps	%xmm3, %xmm7
.LVL462:
.LBE6654:
.LBE6660:
.LBB6661:
.LBB6658:
	subps	%xmm0, %xmm1
.LVL463:
.LBE6658:
.LBE6661:
.LBE6662:
	.loc 1 408 0 discriminator 4
	jne	.L233
	movl	-88(%rbp), %r9d
.LVL464:
.LBE6580:
.LBE6579:
	.loc 1 405 0
	addq	$16, %r8
.LVL465:
	addq	$16, %rdi
.LVL466:
.LBB6673:
.LBB6664:
.LBB6665:
	.loc 4 980 0
	movups	%xmm1, (%rdx)
.LVL467:
.LBE6665:
.LBE6664:
.LBE6673:
	.loc 1 405 0
	addq	$64, %rbx
.LVL468:
	addq	$64, %rdx
.LVL469:
.LBB6674:
.LBB6666:
.LBB6667:
	.loc 4 980 0
	movups	%xmm7, -48(%rdx)
.LVL470:
.LBE6667:
.LBE6666:
.LBE6674:
	.loc 1 405 0
	addl	$16, %r9d
.LBB6675:
.LBB6668:
.LBB6669:
	.loc 4 980 0
	movups	%xmm6, -32(%rdx)
.LVL471:
.LBE6669:
.LBE6668:
.LBB6670:
.LBB6671:
	movups	%xmm8, -16(%rdx)
.LVL472:
.LBE6671:
.LBE6670:
.LBE6675:
	.loc 1 405 0
	cmpl	%r9d, -96(%rbp)
	jle	.L283
	.loc 1 405 0 is_stmt 0 discriminator 1
	movl	76(%r15), %eax
	leal	(%r9,%r14), %ecx
	leal	-15(%rax), %r10d
.LVL473:
	cmpl	%ecx, %r10d
	jg	.L228
	movq	-112(%rbp), %r14
.LVL474:
.L211:
.LBE6578:
.LBB6677:
	.loc 1 442 0 is_stmt 1
	cmpl	%ecx, %eax
	jle	.L277
.LBB6678:
	.loc 1 450 0 discriminator 1
	movslq	%eax, %rbx
	.loc 1 445 0 discriminator 1
	leal	(%rax,%rax), %ecx
	movq	%r15, -112(%rbp)
	.loc 1 450 0 discriminator 1
	leaq	0(,%rbx,4), %r10
	.loc 1 445 0 discriminator 1
	movq	%rbx, -88(%rbp)
	movl	-176(%rbp), %r15d
	movslq	%ecx, %r12
	addl	%eax, %ecx
	movl	-96(%rbp), %ebx
	.loc 1 450 0 discriminator 1
	negq	%r10
	.loc 1 445 0 discriminator 1
	movslq	%ecx, %r13
	jmp	.L254
.LVL475:
	.p2align 4,,10
	.p2align 3
.L284:
	.loc 1 445 0 is_stmt 0
	movzbl	(%r8), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2ss	%ecx, %xmm0
	mulss	80(%r14), %xmm0
	addss	(%rdx), %xmm0
	movss	%xmm0, (%rdx)
	.loc 1 449 0 is_stmt 1
	je	.L218
	.loc 1 450 0
	leaq	(%rdx,%r10), %rcx
	movss	64(%r14), %xmm1
	.loc 1 445 0
	movq	%r8, %r11
	subq	-88(%rbp), %r11
	.loc 1 449 0
	cmpl	$1, %esi
	.loc 1 450 0
	mulss	(%rcx), %xmm1
	subss	%xmm1, %xmm0
	.loc 1 445 0
	pxor	%xmm1, %xmm1
	.loc 1 450 0
	movss	%xmm0, (%rdx)
.LVL476:
	.loc 1 445 0
	movzbl	(%r11), %r11d
	cvtsi2ss	%r11d, %xmm1
	mulss	84(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
	.loc 1 449 0
	je	.L220
	.loc 1 450 0
	movss	68(%r14), %xmm1
	addq	%r10, %rcx
	.loc 1 445 0
	movq	%r8, %r11
	.loc 1 450 0
	mulss	(%rcx), %xmm1
	.loc 1 445 0
	subq	%r12, %r11
	.loc 1 449 0
	cmpl	$2, %esi
	.loc 1 450 0
	subss	%xmm1, %xmm0
	.loc 1 445 0
	pxor	%xmm1, %xmm1
	.loc 1 450 0
	movss	%xmm0, (%rdx)
.LVL477:
	.loc 1 445 0
	movzbl	(%r11), %r11d
	cvtsi2ss	%r11d, %xmm1
	mulss	88(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
	.loc 1 449 0
	je	.L222
	.loc 1 450 0
	movss	72(%r14), %xmm1
	.loc 1 445 0
	movq	%r8, %r11
	.loc 1 450 0
	mulss	(%rcx,%r10), %xmm1
	.loc 1 445 0
	subq	%r13, %r11
	.loc 1 449 0
	cmpl	$3, %esi
	.loc 1 450 0
	subss	%xmm1, %xmm0
	.loc 1 445 0
	pxor	%xmm1, %xmm1
	.loc 1 450 0
	movss	%xmm0, (%rdx)
.LVL478:
	.loc 1 445 0
	movzbl	(%r11), %r11d
	cvtsi2ss	%r11d, %xmm1
	mulss	92(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
	.loc 1 449 0
	je	.L224
	.loc 1 450 0
	movss	(%rcx,%r10,2), %xmm1
	mulss	76(%r14), %xmm1
	subss	%xmm1, %xmm0
.L225:
.LBE6678:
	.loc 1 442 0
	addl	$1, %r9d
.LVL479:
	movss	%xmm0, (%rdx)
.LVL480:
	addq	$1, %r8
.LVL481:
	addq	$1, %rdi
.LVL482:
	addq	$4, %rdx
.LVL483:
	cmpl	%r9d, %ebx
	jle	.L278
	.loc 1 442 0 is_stmt 0 discriminator 1
	leal	(%r9,%r15), %ecx
	cmpl	%ecx, %eax
	jle	.L278
.LVL484:
.L254:
.LBB6679:
	.loc 1 444 0 is_stmt 1 discriminator 1
	testl	%esi, %esi
	jns	.L284
	.loc 1 447 0
	movzbl	(%rdi), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2ss	%ecx, %xmm0
	mulss	80(%r14), %xmm0
	addss	(%rdx), %xmm0
	movss	%xmm0, (%rdx)
.L218:
	.loc 1 452 0
	movzbl	(%rdi), %ecx
	pxor	%xmm1, %xmm1
	cvtsi2ss	%ecx, %xmm1
	mulss	64(%r14), %xmm1
	subss	%xmm1, %xmm0
	.loc 1 447 0
	pxor	%xmm1, %xmm1
	.loc 1 452 0
	movss	%xmm0, (%rdx)
.LVL485:
	.loc 1 447 0
	movzbl	(%rdi), %ecx
	cvtsi2ss	%ecx, %xmm1
	mulss	84(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
.L220:
	.loc 1 452 0
	movzbl	(%rdi), %ecx
	pxor	%xmm1, %xmm1
	cvtsi2ss	%ecx, %xmm1
	mulss	68(%r14), %xmm1
	subss	%xmm1, %xmm0
	.loc 1 447 0
	pxor	%xmm1, %xmm1
	.loc 1 452 0
	movss	%xmm0, (%rdx)
.LVL486:
	.loc 1 447 0
	movzbl	(%rdi), %ecx
	cvtsi2ss	%ecx, %xmm1
	mulss	88(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
.L222:
	.loc 1 452 0
	movzbl	(%rdi), %ecx
	pxor	%xmm1, %xmm1
	cvtsi2ss	%ecx, %xmm1
	mulss	72(%r14), %xmm1
	subss	%xmm1, %xmm0
	.loc 1 447 0
	pxor	%xmm1, %xmm1
	.loc 1 452 0
	movss	%xmm0, (%rdx)
.LVL487:
	.loc 1 447 0
	movzbl	(%rdi), %ecx
	cvtsi2ss	%ecx, %xmm1
	mulss	92(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
.L224:
	.loc 1 452 0
	movzbl	(%rdi), %ecx
	pxor	%xmm1, %xmm1
	cvtsi2ss	%ecx, %xmm1
	mulss	76(%r14), %xmm1
	subss	%xmm1, %xmm0
	jmp	.L225
.LVL488:
	.p2align 4,,10
	.p2align 3
.L231:
.LBE6679:
.LBE6677:
.LBB6680:
.LBB6676:
.LBB6672:
.LBB6663:
	.loc 1 424 0 discriminator 1
	movl	76(%r15), %r10d
.LVL489:
	movq	%rdx, %r11
	imull	%ecx, %r10d
	movslq	%r10d, %r10
	salq	$2, %r10
	subq	%r10, %r11
	movq	%r11, %r10
	jmp	.L234
.LVL490:
	.p2align 4,,10
	.p2align 3
.L230:
	.loc 1 409 0 discriminator 1
	movl	76(%r15), %r10d
	movq	%r8, %r9
	imull	%eax, %r10d
	movslq	%r10d, %r10
	subq	%r10, %r9
	movq	%r9, %r10
	jmp	.L235
.LVL491:
	.p2align 4,,10
	.p2align 3
.L283:
	movq	-112(%rbp), %r14
.LVL492:
.L277:
	movl	72(%r15), %r9d
.LVL493:
.L210:
.LBE6663:
.LBE6672:
.LBE6676:
.LBE6680:
.LBE6573:
	.loc 1 399 0 discriminator 2
	addl	$1, %esi
.LVL494:
	addq	$1, -104(%rbp)
	cmpl	%r9d, %esi
	jl	.L263
.LVL495:
.L213:
.LBE6572:
.LBB6683:
	.loc 1 457 0
	leal	-1(%r9), %edx
	movslq	%edx, %rax
	testl	%edx, %edx
	movl	%edx, %ebx
.LVL496:
	movq	%rax, -200(%rbp)
	js	.L194
	movl	-96(%rbp), %r12d
	movl	-176(%rbp), %r10d
	movq	%rax, %r8
.LVL497:
	.p2align 4,,10
	.p2align 3
.L264:
.LBB6684:
	.loc 1 458 0
	movq	32(%r15), %rax
	movq	%r8, %rsi
	.loc 1 459 0
	movslq	%edx, %rdx
	.loc 1 458 0
	movq	-192(%rbp), %rcx
	.loc 1 460 0
	movq	-184(%rbp), %r11
	movq	16(%rax), %rdi
.LVL498:
.LBB6685:
.LBB6686:
	.loc 2 436 0
	movq	72(%rax), %rax
.LVL499:
	movq	(%rax), %rax
.LVL500:
.LBE6686:
.LBE6685:
	.loc 1 459 0
	imulq	%rax, %rdx
	.loc 1 458 0
	imulq	%rax, %rsi
	addq	%rcx, %rsi
	.loc 1 459 0
	addq	%rdx, %rcx
	.loc 1 460 0
	movq	64(%r15), %rdx
	.loc 1 458 0
	addq	%rdi, %rsi
.LVL501:
	.loc 1 459 0
	addq	%rdi, %rcx
.LVL502:
	.loc 1 460 0
	movq	%r8, %rdi
.LBB6687:
.LBB6688:
	.loc 2 430 0
	movq	72(%rdx), %rax
.LBE6688:
.LBE6687:
	.loc 1 460 0
	imulq	(%rax), %rdi
	movq	%rdi, %rax
	.loc 1 461 0
	movq	48(%r15), %rdi
	.loc 1 460 0
	addq	%r11, %rax
	addq	16(%rdx), %rax
.LVL503:
.LBB6689:
.LBB6690:
	.loc 2 430 0
	movq	72(%rdi), %rdx
.LBE6690:
.LBE6689:
	.loc 1 461 0
	imulq	(%rdx), %r8
	movq	%r8, %rdx
	addq	%r11, %rdx
	addq	16(%rdi), %rdx
.LVL504:
.LBB6691:
	.loc 1 463 0
	testl	%r12d, %r12d
	jle	.L191
	movl	76(%r15), %r8d
	leal	-15(%r8), %edi
.LVL505:
	cmpl	%edi, -172(%rbp)
	jge	.L248
	movq	-224(%rbp), %r13
	xorl	%edi, %edi
.LVL506:
	.p2align 4,,10
	.p2align 3
.L193:
	xorl	%r8d, %r8d
	pxor	%xmm0, %xmm0
.LBB6692:
.LBB6693:
.LBB6694:
	.loc 1 467 0
	addl	$1, %r8d
	movq	%r13, %r11
	subl	%r8d, %r9d
	movdqa	(%r15), %xmm4
.LVL507:
	cmpl	%ebx, %r9d
.LBE6694:
.LBE6693:
.LBE6692:
	.loc 1 463 0
	movaps	%xmm0, %xmm7
.LVL508:
	movaps	%xmm0, %xmm6
.LVL509:
	movaps	%xmm0, %xmm3
.LVL510:
.LBB6860:
.LBB6827:
.LBB6823:
	.loc 1 467 0
	jg	.L208
.LVL511:
.L285:
.LBB6695:
	.loc 1 496 0
	movss	(%r11), %xmm1
	addq	$4, %r11
.LBB6696:
.LBB6697:
	.loc 3 698 0
	movdqu	(%rcx), %xmm2
.LBE6697:
.LBE6696:
	.loc 1 496 0
	subss	-36(%r11), %xmm1
.LVL512:
.LBE6695:
.LBE6823:
	.loc 1 466 0
	cmpl	$4, %r8d
.LBB6824:
.LBB6738:
.LBB6698:
.LBB6699:
	.loc 3 965 0
	movdqa	%xmm2, %xmm5
.LBE6699:
.LBE6698:
.LBB6701:
.LBB6702:
	.loc 3 989 0
	punpcklbw	%xmm4, %xmm2
.LBE6702:
.LBE6701:
.LBB6703:
.LBB6700:
	.loc 3 965 0
	punpckhbw	%xmm4, %xmm5
.LBE6700:
.LBE6703:
.LBB6704:
.LBB6705:
	.loc 3 971 0
	movdqa	%xmm2, %xmm8
.LBE6705:
.LBE6704:
.LBB6707:
.LBB6708:
	.loc 3 995 0
	punpcklwd	%xmm4, %xmm2
.LBE6708:
.LBE6707:
.LBB6709:
.LBB6710:
	.loc 4 891 0
	shufps	$0, %xmm1, %xmm1
.LVL513:
.LBE6710:
.LBE6709:
.LBB6711:
.LBB6712:
	.loc 3 995 0
	movdqa	%xmm5, %xmm9
.LBE6712:
.LBE6711:
.LBB6714:
.LBB6715:
	.loc 3 971 0
	punpckhwd	%xmm4, %xmm5
.LVL514:
.LBE6715:
.LBE6714:
.LBB6716:
.LBB6717:
	.loc 3 767 0
	cvtdq2ps	%xmm2, %xmm2
.LBE6717:
.LBE6716:
.LBB6718:
.LBB6706:
	.loc 3 971 0
	punpckhwd	%xmm4, %xmm8
.LVL515:
.LBE6706:
.LBE6718:
.LBB6719:
.LBB6713:
	.loc 3 995 0
	punpcklwd	%xmm4, %xmm9
.LVL516:
.LBE6713:
.LBE6719:
.LBB6720:
.LBB6721:
	.loc 3 767 0
	cvtdq2ps	%xmm5, %xmm5
.LVL517:
.LBE6721:
.LBE6720:
.LBB6722:
.LBB6723:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
	addps	%xmm5, %xmm3
.LVL518:
.LBE6723:
.LBE6722:
.LBB6724:
.LBB6725:
	.loc 3 767 0
	cvtdq2ps	%xmm9, %xmm5
.LVL519:
.LBE6725:
.LBE6724:
.LBB6726:
.LBB6727:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
	addps	%xmm5, %xmm6
.LVL520:
.LBE6727:
.LBE6726:
.LBB6728:
.LBB6729:
	.loc 3 767 0
	cvtdq2ps	%xmm8, %xmm5
.LVL521:
.LBE6729:
.LBE6728:
.LBB6730:
.LBB6731:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
.LBE6731:
.LBE6730:
.LBB6733:
.LBB6734:
	mulps	%xmm2, %xmm1
.LBE6734:
.LBE6733:
.LBB6736:
.LBB6732:
	addps	%xmm5, %xmm7
.LVL522:
.LBE6732:
.LBE6736:
.LBB6737:
.LBB6735:
	addps	%xmm1, %xmm0
.LVL523:
.LBE6735:
.LBE6737:
.LBE6738:
.LBE6824:
	.loc 1 466 0
	je	.L205
.LVL524:
.L286:
	movl	72(%r15), %r9d
.LBB6825:
	.loc 1 467 0
	addl	$1, %r8d
.LVL525:
	subl	%r8d, %r9d
	cmpl	%ebx, %r9d
	jle	.L285
.LVL526:
.L208:
.LBB6739:
.LBB6740:
.LBB6741:
	.loc 3 698 0
	movl	76(%r15), %r9d
.LBE6741:
.LBE6740:
.LBB6743:
.LBB6744:
.LBB6745:
.LBB6746:
	.loc 4 884 0
	movss	(%r11), %xmm1
.LVL527:
	addq	$4, %r11
.LBE6746:
.LBE6745:
.LBE6744:
.LBE6743:
.LBB6747:
.LBB6748:
	.loc 4 743 0
	shufps	$0, %xmm1, %xmm1
.LVL528:
.LBE6748:
.LBE6747:
.LBB6749:
.LBB6742:
	.loc 3 698 0
	imull	%r8d, %r9d
	movslq	%r9d, %r9
	movdqu	(%rsi,%r9), %xmm2
.LVL529:
.LBE6742:
.LBE6749:
.LBB6750:
.LBB6751:
	.loc 3 965 0
	movdqa	%xmm2, %xmm5
.LBE6751:
.LBE6750:
.LBB6753:
.LBB6754:
	.loc 3 989 0
	punpcklbw	%xmm4, %xmm2
.LVL530:
.LBE6754:
.LBE6753:
.LBB6755:
.LBB6752:
	.loc 3 965 0
	punpckhbw	%xmm4, %xmm5
.LVL531:
.LBE6752:
.LBE6755:
.LBB6756:
.LBB6757:
	.loc 3 971 0
	movdqa	%xmm2, %xmm8
.LBE6757:
.LBE6756:
.LBB6759:
.LBB6760:
	.loc 3 995 0
	movdqa	%xmm5, %xmm9
.LBE6760:
.LBE6759:
.LBB6762:
.LBB6763:
	.loc 3 971 0
	punpckhwd	%xmm4, %xmm5
.LVL532:
.LBE6763:
.LBE6762:
.LBB6764:
.LBB6758:
	punpckhwd	%xmm4, %xmm8
.LVL533:
.LBE6758:
.LBE6764:
.LBB6765:
.LBB6761:
	.loc 3 995 0
	punpcklwd	%xmm4, %xmm9
.LBE6761:
.LBE6765:
.LBB6766:
.LBB6767:
	.loc 3 767 0
	cvtdq2ps	%xmm5, %xmm5
.LBE6767:
.LBE6766:
.LBB6768:
.LBB6769:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
.LBE6769:
.LBE6768:
.LBB6771:
.LBB6772:
	.loc 3 995 0
	punpcklwd	%xmm4, %xmm2
.LVL534:
.LBE6772:
.LBE6771:
.LBB6773:
.LBB6770:
	.loc 4 183 0
	addps	%xmm5, %xmm3
.LVL535:
.LBE6770:
.LBE6773:
.LBB6774:
.LBB6775:
	.loc 3 767 0
	cvtdq2ps	%xmm9, %xmm5
.LVL536:
.LBE6775:
.LBE6774:
.LBB6776:
.LBB6777:
	cvtdq2ps	%xmm2, %xmm2
.LVL537:
.LBE6777:
.LBE6776:
.LBB6778:
.LBB6779:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
.LBE6779:
.LBE6778:
.LBB6781:
.LBB6782:
	.loc 4 931 0
	movups	48(%rax,%r9,4), %xmm9
.LVL538:
.LBE6782:
.LBE6781:
.LBB6783:
.LBB6780:
	.loc 4 183 0
	addps	%xmm5, %xmm6
.LVL539:
.LBE6780:
.LBE6783:
.LBB6784:
.LBB6785:
	.loc 3 767 0
	cvtdq2ps	%xmm8, %xmm5
.LVL540:
.LBE6785:
.LBE6784:
.LBB6786:
.LBB6787:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
.LBE6787:
.LBE6786:
.LBB6789:
.LBB6790:
	.loc 4 931 0
	movups	32(%rax,%r9,4), %xmm8
.LVL541:
.LBE6790:
.LBE6789:
.LBB6791:
.LBB6792:
	.loc 4 183 0
	mulps	%xmm2, %xmm1
.LVL542:
.LBE6792:
.LBE6791:
.LBB6794:
.LBB6795:
	.loc 4 931 0
	movups	(%rax,%r9,4), %xmm2
.LBE6795:
.LBE6794:
.LBB6796:
.LBB6788:
	.loc 4 183 0
	addps	%xmm5, %xmm7
.LVL543:
.LBE6788:
.LBE6796:
.LBB6797:
.LBB6798:
	.loc 4 931 0
	movups	16(%rax,%r9,4), %xmm5
.LBE6798:
.LBE6797:
.LBB6799:
.LBB6793:
	.loc 4 183 0
	addps	%xmm1, %xmm0
.LVL544:
.LBE6793:
.LBE6799:
.LBB6800:
.LBB6801:
.LBB6802:
.LBB6803:
	.loc 4 884 0
	movss	-36(%r11), %xmm1
.LVL545:
.LBE6803:
.LBE6802:
.LBE6801:
.LBE6800:
.LBE6739:
.LBE6825:
	.loc 1 466 0
	cmpl	$4, %r8d
.LBB6826:
.LBB6822:
.LBB6804:
.LBB6805:
	.loc 4 743 0
	shufps	$0, %xmm1, %xmm1
.LVL546:
.LBE6805:
.LBE6804:
.LBB6806:
.LBB6807:
	.loc 4 189 0
	mulps	%xmm1, %xmm9
.LVL547:
.LBE6807:
.LBE6806:
.LBB6809:
.LBB6810:
	mulps	%xmm1, %xmm8
.LVL548:
.LBE6810:
.LBE6809:
.LBB6812:
.LBB6813:
	mulps	%xmm1, %xmm5
.LVL549:
.LBE6813:
.LBE6812:
.LBB6815:
.LBB6808:
	subps	%xmm9, %xmm3
.LBE6808:
.LBE6815:
.LBB6816:
.LBB6817:
	mulps	%xmm2, %xmm1
.LVL550:
.LBE6817:
.LBE6816:
.LBB6819:
.LBB6811:
	subps	%xmm8, %xmm6
.LBE6811:
.LBE6819:
.LBB6820:
.LBB6814:
	subps	%xmm5, %xmm7
.LBE6814:
.LBE6820:
.LBB6821:
.LBB6818:
	subps	%xmm1, %xmm0
.LVL551:
.LBE6818:
.LBE6821:
.LBE6822:
.LBE6826:
	.loc 1 466 0
	jne	.L286
.LVL552:
.L205:
.LBE6827:
.LBB6828:
.LBB6829:
	.loc 4 980 0
	movups	%xmm0, (%rax)
.LVL553:
.LBE6829:
.LBE6828:
.LBE6860:
	.loc 1 463 0
	addl	$16, %edi
.LVL554:
	addq	$16, %rsi
.LVL555:
	addq	$16, %rcx
.LVL556:
.LBB6861:
.LBB6830:
.LBB6831:
	.loc 4 980 0
	movups	%xmm7, 16(%rax)
.LVL557:
.LBE6831:
.LBE6830:
.LBE6861:
	.loc 1 463 0
	addq	$64, %rdx
.LVL558:
	addq	$64, %rax
.LVL559:
.LBB6862:
.LBB6832:
.LBB6833:
	.loc 4 980 0
	movups	%xmm6, -32(%rax)
.LVL560:
.LBE6833:
.LBE6832:
.LBB6834:
.LBB6835:
	movups	%xmm3, -16(%rax)
.LVL561:
.LBE6835:
.LBE6834:
.LBB6836:
.LBB6837:
	.loc 4 931 0
	movups	-16(%rdx), %xmm1
.LVL562:
.LBE6837:
.LBE6836:
.LBB6838:
.LBB6839:
	movups	-32(%rdx), %xmm2
.LVL563:
.LBE6839:
.LBE6838:
.LBB6840:
.LBB6841:
	.loc 4 980 0
	addps	%xmm1, %xmm3
.LBE6841:
.LBE6840:
.LBB6843:
.LBB6844:
	.loc 4 931 0
	movups	-48(%rdx), %xmm4
.LVL564:
.LBE6844:
.LBE6843:
.LBB6845:
.LBB6846:
	.loc 4 980 0
	addps	%xmm2, %xmm6
.LBE6846:
.LBE6845:
.LBB6848:
.LBB6849:
	.loc 4 931 0
	movups	-64(%rdx), %xmm5
.LVL565:
.LBE6849:
.LBE6848:
.LBB6850:
.LBB6851:
	.loc 4 980 0
	addps	%xmm4, %xmm7
.LBE6851:
.LBE6850:
.LBB6853:
.LBB6842:
	movups	%xmm3, -16(%rdx)
.LBE6842:
.LBE6853:
.LBB6854:
.LBB6855:
	addps	%xmm5, %xmm0
.LBE6855:
.LBE6854:
.LBB6857:
.LBB6847:
	movups	%xmm6, -32(%rdx)
.LBE6847:
.LBE6857:
.LBB6858:
.LBB6852:
	movups	%xmm7, -48(%rdx)
.LBE6852:
.LBE6858:
.LBB6859:
.LBB6856:
	movups	%xmm0, -64(%rdx)
.LVL566:
.LBE6856:
.LBE6859:
.LBE6862:
	.loc 1 463 0
	cmpl	%edi, %r12d
	jle	.L191
	leal	(%rdi,%r10), %r9d
	.loc 1 463 0 is_stmt 0 discriminator 1
	movl	76(%r15), %r8d
.LVL567:
	leal	-15(%r8), %r11d
	cmpl	%r9d, %r11d
	jle	.L192
	movl	72(%r15), %r9d
	jmp	.L193
.LVL568:
	.p2align 4,,10
	.p2align 3
.L273:
	movq	-208(%rbp), %r15
.LVL569:
.L191:
.LBE6691:
.LBE6684:
	.loc 1 457 0 is_stmt 1
	subl	$1, %ebx
.LVL570:
	subq	$1, -200(%rbp)
	cmpl	$-1, %ebx
	je	.L194
	movl	72(%r15), %r9d
	movq	-200(%rbp), %r8
	leal	-1(%r9), %edx
.LVL571:
	jmp	.L264
.LVL572:
.L248:
.LBB6870:
.LBB6863:
	.loc 1 463 0
	movl	-172(%rbp), %r9d
.LBE6863:
	.loc 1 462 0
	xorl	%edi, %edi
.LVL573:
	.p2align 4,,10
	.p2align 3
.L192:
.LBB6864:
	.loc 1 525 0
	cmpl	%r8d, %r9d
	jge	.L191
	movl	72(%r15), %r9d
	movq	%r15, -208(%rbp)
	movl	%r8d, -112(%rbp)
	leal	-1(%r9), %r11d
	leal	-2(%r9), %r13d
	movl	%r11d, -88(%rbp)
	leal	-3(%r9), %r11d
	movl	%r11d, -104(%rbp)
	leal	-4(%r9), %r11d
	movl	%r11d, -212(%rbp)
.LBB6865:
.LBB6866:
	.loc 1 528 0
	movslq	%r8d, %r11
	movl	-212(%rbp), %r15d
	leaq	0(,%r11,4), %r9
	movq	%r11, -120(%rbp)
	movq	%r9, -128(%rbp)
	leal	(%r8,%r8), %r9d
	movslq	%r9d, %r11
	addl	%r8d, %r9d
	movq	%r11, -136(%rbp)
	salq	$2, %r11
	movq	%r11, -144(%rbp)
	movslq	%r9d, %r11
	leaq	0(,%r11,4), %r9
	movq	%r11, -152(%rbp)
	leal	0(,%r8,4), %r11d
	movslq	%r11d, %r11
	movq	%r9, -160(%rbp)
	leaq	0(,%r11,4), %r9
	addq	%r11, %rsi
.LVL574:
	movq	%r9, -168(%rbp)
	jmp	.L245
.LVL575:
	.p2align 4,,10
	.p2align 3
.L287:
	movq	-120(%rbp), %r8
	movq	%rsi, %r9
	subq	%r11, %r9
	pxor	%xmm2, %xmm2
	movss	64(%r14), %xmm0
	.loc 1 527 0
	cmpl	%r13d, %ebx
	.loc 1 528 0
	movzbl	(%r9,%r8), %r9d
	movq	-128(%rbp), %r8
	movss	(%rax), %xmm3
	mulss	(%rax,%r8), %xmm0
	cvtsi2ss	%r9d, %xmm2
	mulss	96(%r14), %xmm2
	subss	%xmm0, %xmm2
	addss	%xmm2, %xmm3
	movss	%xmm3, (%rax)
.LVL576:
	.loc 1 527 0
	jge	.L198
.L288:
	.loc 1 528 0
	movq	-136(%rbp), %r8
	movq	%rsi, %r9
	subq	%r11, %r9
	pxor	%xmm1, %xmm1
	movss	68(%r14), %xmm0
	.loc 1 527 0
	cmpl	-104(%rbp), %ebx
	.loc 1 528 0
	movzbl	(%r9,%r8), %r9d
	movq	-144(%rbp), %r8
	mulss	(%rax,%r8), %xmm0
	cvtsi2ss	%r9d, %xmm1
	mulss	100(%r14), %xmm1
	subss	%xmm0, %xmm1
	movaps	%xmm1, %xmm2
	addss	%xmm3, %xmm2
	movss	%xmm2, (%rax)
.LVL577:
	.loc 1 527 0
	jge	.L200
.L289:
	.loc 1 528 0
	movq	-152(%rbp), %r8
	movq	%rsi, %r9
	subq	%r11, %r9
	pxor	%xmm0, %xmm0
	movss	72(%r14), %xmm1
	.loc 1 527 0
	cmpl	%r15d, %ebx
	.loc 1 528 0
	movzbl	(%r9,%r8), %r9d
	movq	-160(%rbp), %r8
	mulss	(%rax,%r8), %xmm1
	cvtsi2ss	%r9d, %xmm0
	mulss	104(%r14), %xmm0
	subss	%xmm1, %xmm0
	movaps	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	movss	%xmm1, (%rax)
.LVL578:
	.loc 1 527 0
	jge	.L202
.L290:
	.loc 1 528 0
	movzbl	(%rsi), %r9d
	pxor	%xmm0, %xmm0
	movq	-168(%rbp), %r8
	movss	76(%r14), %xmm2
	cvtsi2ss	%r9d, %xmm0
	mulss	(%rax,%r8), %xmm2
	mulss	108(%r14), %xmm0
	subss	%xmm2, %xmm0
	addss	%xmm1, %xmm0
.L203:
	movss	%xmm0, (%rax)
.LVL579:
.LBE6866:
.LBE6865:
	.loc 1 525 0
	addl	$1, %edi
.LVL580:
	addq	$1, %rcx
.LVL581:
	addq	$4, %rax
.LVL582:
	addq	$4, %rdx
.LVL583:
.LBB6868:
	.loc 1 533 0
	addss	-4(%rdx), %xmm0
	movss	%xmm0, -4(%rdx)
.LVL584:
.LBE6868:
	.loc 1 525 0
	cmpl	%edi, %r12d
	jle	.L273
	addq	$1, %rsi
.LVL585:
	.loc 1 525 0 is_stmt 0 discriminator 1
	leal	(%rdi,%r10), %r9d
	cmpl	-112(%rbp), %r9d
	jge	.L273
.LVL586:
.L245:
.LBB6869:
.LBB6867:
	.loc 1 527 0 is_stmt 1 discriminator 1
	cmpl	-88(%rbp), %ebx
	jl	.L287
	.loc 1 530 0
	movzbl	(%rcx), %r9d
	pxor	%xmm0, %xmm0
	movss	96(%r14), %xmm2
	.loc 1 527 0
	cmpl	%r13d, %ebx
	.loc 1 530 0
	subss	64(%r14), %xmm2
	movss	(%rax), %xmm3
	cvtsi2ss	%r9d, %xmm0
	mulss	%xmm0, %xmm2
	addss	%xmm2, %xmm3
	movss	%xmm3, (%rax)
.LVL587:
	.loc 1 527 0
	jl	.L288
.L198:
	.loc 1 530 0
	movzbl	(%rcx), %r9d
	pxor	%xmm0, %xmm0
	movss	100(%r14), %xmm1
	subss	68(%r14), %xmm1
	.loc 1 527 0
	cmpl	-104(%rbp), %ebx
	.loc 1 530 0
	cvtsi2ss	%r9d, %xmm0
	mulss	%xmm0, %xmm1
	movaps	%xmm1, %xmm2
	addss	%xmm3, %xmm2
	movss	%xmm2, (%rax)
.LVL588:
	.loc 1 527 0
	jl	.L289
.L200:
	.loc 1 530 0
	movzbl	(%rcx), %r9d
	pxor	%xmm3, %xmm3
	movss	104(%r14), %xmm0
	.loc 1 527 0
	cmpl	%r15d, %ebx
	.loc 1 530 0
	subss	72(%r14), %xmm0
	cvtsi2ss	%r9d, %xmm3
	mulss	%xmm3, %xmm0
	movaps	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	movss	%xmm1, (%rax)
.LVL589:
	.loc 1 527 0
	jl	.L290
.L202:
	.loc 1 530 0
	movzbl	(%rcx), %r9d
	pxor	%xmm2, %xmm2
	movss	108(%r14), %xmm0
	subss	76(%r14), %xmm0
	cvtsi2ss	%r9d, %xmm2
	mulss	%xmm2, %xmm0
	addss	%xmm1, %xmm0
	jmp	.L203
.LVL590:
	.p2align 4,,10
	.p2align 3
.L278:
	movq	-112(%rbp), %r15
	movl	72(%r15), %r9d
.LVL591:
	jmp	.L210
.LVL592:
.L194:
	movl	-96(%rbp), %ebx
.LVL593:
	addl	%ebx, -172(%rbp)
	movq	-248(%rbp), %rdi
	movl	-172(%rbp), %eax
.LVL594:
	addq	%rdi, -192(%rbp)
	addl	%ebx, -176(%rbp)
	movq	-240(%rbp), %rdi
	addq	%rdi, -184(%rbp)
	cmpl	%eax, -216(%rbp)
	jg	.L189
	leaq	-64(%rbp), %rsi
	leaq	-72(%rbp), %rdi
	movq	%r14, %r13
	movq	%r15, %r14
	call	GOMP_loop_dynamic_next
.LVL595:
	testb	%al, %al
	jne	.L190
	movq	%r15, %r10
	movq	%r13, %r14
.LVL596:
.L186:
	movq	%r10, -88(%rbp)
	call	GOMP_loop_end
.LVL597:
.LBE6867:
.LBE6869:
.LBE6864:
.LBE6870:
.LBE6683:
.LBE6571:
.LBE6570:
.LBB6873:
	.loc 1 539 0
	movq	-88(%rbp), %r10
	movl	72(%r10), %eax
	movq	%r10, -96(%rbp)
	movl	%eax, %r15d
	movl	%eax, -184(%rbp)
	addl	$3, %eax
	testl	%r15d, %r15d
	cmovns	%r15d, %eax
	sarl	$2, %eax
	movl	%eax, %ebx
	call	omp_get_num_threads
.LVL598:
	movl	%eax, %r12d
	call	omp_get_thread_num
.LVL599:
	movl	%ebx, %edi
	imull	%eax, %edi
	leal	(%rdi,%rbx), %edx
	movl	%edi, %esi
	movl	%edi, -88(%rbp)
	cmpl	%r15d, %edx
	cmovg	%r15d, %edx
	cmpl	%r15d, %esi
	movl	%edx, -152(%rbp)
	jge	.L236
	movslq	%eax, %rdi
	addl	%r12d, %eax
	movq	-96(%rbp), %r10
	movq	%rdi, -160(%rbp)
	movl	%ebx, %edi
	imull	%r12d, %edi
	movq	%r10, %r15
	movl	%edi, -172(%rbp)
	movl	%eax, %edi
	addl	$1, %eax
	imull	%ebx, %edi
	imull	%ebx, %eax
	movl	%edi, -168(%rbp)
	subl	%edi, %eax
	movl	%eax, -192(%rbp)
.L240:
	movslq	-88(%rbp), %r8
	movq	%r15, %rcx
.L239:
.LVL600:
.LBB6874:
.LBB6875:
	.loc 1 543 0
	movl	76(%rcx), %eax
	.loc 1 605 0
	movq	%rsp, -96(%rbp)
.LVL601:
.LBB6876:
.LBB6877:
	.loc 2 430 0
	movq	%r8, %rdi
.LBE6877:
.LBE6876:
.LBB6882:
.LBB6883:
	movq	-160(%rbp), %r10
.LBE6883:
.LBE6882:
.LBB6885:
.LBB6886:
	movq	%r8, %r13
.LBE6886:
.LBE6885:
.LBB6891:
.LBB6892:
	.loc 5 90 0
	xorl	%esi, %esi
.LBE6892:
.LBE6891:
.LBB6896:
.LBB6887:
	.loc 2 430 0
	movq	%r8, -144(%rbp)
.LBE6887:
.LBE6896:
.LBB6897:
.LBB6893:
	.loc 5 90 0
	movq	%rcx, -112(%rbp)
.LBE6893:
.LBE6897:
	.loc 1 543 0
	addl	$8, %eax
	cltq
	leaq	18(,%rax,4), %rax
	andq	$-16, %rax
	subq	%rax, %rsp
	.loc 1 545 0
	movq	48(%rcx), %rax
	.loc 1 543 0
	leaq	3(%rsp), %r9
.LBB6898:
.LBB6878:
	.loc 2 430 0
	movq	72(%rax), %rdx
.LBE6878:
.LBE6898:
	.loc 1 543 0
	shrq	$2, %r9
	leaq	0(,%r9,4), %r12
.LVL602:
	movq	%r9, -136(%rbp)
.LBB6899:
.LBB6879:
	.loc 2 430 0
	imulq	(%rdx), %rdi
.LBE6879:
.LBE6899:
	.loc 1 552 0
	addq	$16, %r12
.LVL603:
.LBB6900:
.LBB6880:
	.loc 2 430 0
	movq	%rdi, %rdx
	addq	16(%rax), %rdx
.LBE6880:
.LBE6900:
	.loc 1 546 0
	movq	56(%rcx), %rax
.LVL604:
.LBB6901:
.LBB6881:
	.loc 2 430 0
	movq	%rdx, -104(%rbp)
.LBE6881:
.LBE6901:
.LBB6902:
.LBB6884:
	movq	72(%rax), %rdx
	imulq	(%rdx), %r10
	addq	16(%rax), %r10
.LVL605:
.LBE6884:
.LBE6902:
	.loc 1 547 0
	movq	40(%rcx), %rax
.LVL606:
.LBB6903:
.LBB6888:
	.loc 2 430 0
	movq	72(%rax), %rdx
.LBE6888:
.LBE6903:
	.loc 1 551 0
	leaq	16(%r10), %rbx
	movq	%r10, -128(%rbp)
.LBB6904:
.LBB6889:
	.loc 2 430 0
	imulq	(%rdx), %r13
.LBE6889:
.LBE6904:
.LBB6905:
.LBB6894:
	.loc 5 90 0
	movq	%rbx, %rdi
.LBE6894:
.LBE6905:
.LBB6906:
.LBB6890:
	.loc 2 430 0
	addq	16(%rax), %r13
.LVL607:
	movl	0(,%r9,4), %eax
.LVL608:
.LBE6890:
.LBE6906:
.LBB6907:
.LBB6908:
	.loc 5 53 0
	movl	%eax, (%r10)
.LVL609:
.LBE6908:
.LBE6907:
.LBB6909:
.LBB6910:
	movl	%eax, 4(%r10)
.LVL610:
.LBE6910:
.LBE6909:
.LBB6911:
.LBB6912:
	movq	(%r10), %rax
	movq	%rax, 8(%r10)
.LVL611:
.LBE6912:
.LBE6911:
.LBB6913:
.LBB6895:
	.loc 5 90 0
	movslq	76(%rcx), %rdx
	salq	$2, %rdx
.LVL612:
	call	memset
.LVL613:
.LBE6895:
.LBE6913:
	.loc 1 552 0
	movq	-112(%rbp), %rcx
.LBB6914:
.LBB6915:
	.loc 5 53 0
	movq	-104(%rbp), %rsi
	movq	%r12, %rdi
.LBE6915:
.LBE6914:
	.loc 1 552 0
	movslq	76(%rcx), %r15
	movq	%rcx, -120(%rbp)
.LBB6918:
.LBB6916:
	.loc 5 53 0
	leaq	0(,%r15,4), %rdx
.LBE6916:
.LBE6918:
	.loc 1 552 0
	movl	%r15d, -112(%rbp)
.LVL614:
.LBB6919:
.LBB6917:
	.loc 5 53 0
	call	memcpy
.LVL615:
.LBE6917:
.LBE6919:
.LBB6920:
.LBB6921:
	movq	-128(%rbp), %r10
.LBE6921:
.LBE6920:
.LBB6925:
	.loc 1 556 0
	movl	-112(%rbp), %r11d
.LBE6925:
.LBB6960:
.LBB6922:
	.loc 5 53 0
	movq	-136(%rbp), %r9
.LBE6922:
.LBE6960:
.LBB6961:
	.loc 1 556 0
	movq	-120(%rbp), %rcx
	movq	-144(%rbp), %r8
.LBE6961:
.LBB6962:
.LBB6923:
	.loc 5 53 0
	movq	(%r10), %rax
	movq	8(%r10), %rdx
.LBE6923:
.LBE6962:
.LBB6963:
	.loc 1 556 0
	testl	%r11d, %r11d
.LBE6963:
.LBB6964:
.LBB6924:
	.loc 5 53 0
	movq	%rax, 0(,%r9,4)
	movq	%rdx, 8(,%r9,4)
.LVL616:
.LBE6924:
.LBE6964:
.LBB6965:
	.loc 1 556 0
	jle	.L291
	movss	4(,%r9,4), %xmm0
	.loc 1 556 0 is_stmt 0 discriminator 2
	xorl	%esi, %esi
	movss	8(,%r9,4), %xmm3
	movss	12(,%r9,4), %xmm2
.LVL617:
	.p2align 4,,10
	.p2align 3
.L243:
.LBB6926:
.LBB6927:
.LBB6928:
	.loc 4 946 0 is_stmt 1 discriminator 2
	movaps	%xmm3, %xmm1
.LBE6928:
.LBE6927:
	.loc 1 567 0 discriminator 2
	movss	(%r12), %xmm4
.LBB6931:
.LBB6932:
	.loc 4 946 0 discriminator 2
	movss	-12(%rbx), %xmm5
.LBE6932:
.LBE6931:
.LBB6936:
.LBB6929:
	unpcklps	%xmm0, %xmm1
	movaps	%xmm4, %xmm0
.LBE6929:
.LBE6936:
.LBB6937:
.LBB6933:
	insertps	$0x10, -16(%rbx), %xmm5
.LBE6933:
.LBE6937:
.LBB6938:
.LBB6930:
	unpcklps	%xmm2, %xmm0
	movlhps	%xmm1, %xmm0
.LBE6930:
.LBE6938:
.LBB6939:
.LBB6934:
	movss	-4(%rbx), %xmm1
	insertps	$0x10, -8(%rbx), %xmm1
.LBE6934:
.LBE6939:
.LBB6940:
.LBB6941:
	.loc 4 189 0 discriminator 2
	mulps	32(%r14), %xmm0
.LBE6941:
.LBE6940:
.LBB6943:
.LBB6935:
	.loc 4 946 0 discriminator 2
	movlhps	%xmm5, %xmm1
.LBE6935:
.LBE6943:
.LBB6944:
.LBB6942:
	.loc 4 189 0 discriminator 2
	mulps	16(%r14), %xmm1
	subps	%xmm1, %xmm0
.LBE6942:
.LBE6944:
.LBB6945:
.LBB6946:
	.file 7 "/usr/lib/gcc/x86_64-linux-gnu/5/include/pmmintrin.h"
	.loc 7 58 0 discriminator 2
	haddps	%xmm0, %xmm0
.LBE6946:
.LBE6945:
.LBB6947:
.LBB6948:
	haddps	%xmm0, %xmm0
.LBE6948:
.LBE6947:
.LBB6949:
.LBB6950:
	.loc 4 960 0 discriminator 2
	movss	%xmm0, (%rbx)
.LBE6950:
.LBE6949:
.LBB6951:
.LBB6952:
.LBB6953:
.LBB6954:
.LBB6955:
.LBB6956:
	.loc 3 63 0 discriminator 2
	cvtss2sd	%xmm0, %xmm0
.LBE6956:
.LBE6955:
.LBB6957:
.LBB6958:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm0, %edx
	movaps	%xmm3, %xmm0
	movaps	%xmm2, %xmm3
	movaps	%xmm4, %xmm2
.LBE6958:
.LBE6957:
.LBE6954:
.LBE6953:
.LBE6952:
.LBE6951:
	.loc 1 575 0 discriminator 2
	testl	%edx, %edx
	setg	%al
	negl	%eax
	cmpl	$255, %edx
	cmovbe	%edx, %eax
.LBE6926:
	.loc 1 556 0 discriminator 2
	addl	$1, %esi
.LVL618:
	addq	$4, %r12
.LVL619:
.LBB6959:
	.loc 1 575 0 discriminator 2
	movb	%al, 0(%r13)
.LBE6959:
	.loc 1 556 0 discriminator 2
	movslq	76(%rcx), %rax
	addq	$4, %rbx
.LVL620:
	addq	$1, %r13
.LVL621:
	cmpl	%esi, %eax
	jg	.L243
.LVL622:
.L244:
.LBE6965:
	.loc 1 577 0
	movq	40(%rcx), %rdx
.LVL623:
	movq	%r8, %rdi
	movq	%r8, -112(%rbp)
	.loc 1 581 0
	movq	%rcx, -104(%rbp)
	.loc 1 578 0
	leaq	-4(%r12), %r13
.LVL624:
.LBB6966:
.LBB6967:
	.loc 2 430 0
	movq	72(%rdx), %rsi
.LBE6967:
.LBE6966:
	.loc 1 577 0
	imulq	(%rsi), %rdi
.LBB6968:
.LBB6969:
	.loc 5 90 0
	xorl	%esi, %esi
.LBE6969:
.LBE6968:
	.loc 1 577 0
	leaq	-1(%rax,%rdi), %r15
	movl	-4(%r12), %eax
	addq	16(%rdx), %r15
.LVL625:
.LBB6972:
.LBB6970:
	.loc 5 90 0
	movq	%rbx, %rdi
.LBE6970:
.LBE6972:
	.loc 1 584 0
	subq	$4, %rbx
.LVL626:
.LBB6973:
.LBB6974:
	.loc 5 53 0
	movl	%eax, 4(%rbx)
.LVL627:
.LBE6974:
.LBE6973:
.LBB6975:
.LBB6976:
	movl	%eax, 8(%rbx)
.LVL628:
.LBE6976:
.LBE6975:
.LBB6977:
.LBB6978:
	movq	4(%rbx), %rax
	movq	%rax, 12(%rbx)
.LVL629:
.LBE6978:
.LBE6977:
	.loc 1 581 0
	movslq	76(%rcx), %rdx
.LVL630:
	salq	$2, %rdx
.LVL631:
.LBB6979:
.LBB6971:
	.loc 5 90 0
	subq	%rdx, %rdi
.LVL632:
	call	memset
.LVL633:
	movq	4(%rbx), %rax
	movq	12(%rbx), %rdx
.LBE6971:
.LBE6979:
.LBB6980:
	.loc 1 585 0
	movq	-104(%rbp), %rcx
	movq	-112(%rbp), %r8
.LBE6980:
.LBB7021:
.LBB7022:
	.loc 5 53 0
	movq	%rax, (%r12)
	movq	%rdx, 8(%r12)
.LVL634:
.LBE7022:
.LBE7021:
.LBB7023:
	.loc 1 585 0
	movl	76(%rcx), %eax
.LVL635:
	testl	%eax, %eax
	jle	.L241
	subl	$1, %eax
.LVL636:
	movq	$-8, %rdi
	salq	$2, %rax
	subq	%rax, %rdi
	addq	%rdi, %r12
.LVL637:
	.p2align 4,,10
	.p2align 3
.L242:
.LBB6981:
.LBB6982:
.LBB6983:
	.loc 4 946 0 discriminator 2
	movss	12(%r13), %xmm1
	movss	4(%r13), %xmm0
	insertps	$0x10, 16(%r13), %xmm1
.LBE6983:
.LBE6982:
.LBB6986:
.LBB6987:
	movss	12(%rbx), %xmm2
.LBE6987:
.LBE6986:
.LBB6991:
.LBB6984:
	insertps	$0x10, 8(%r13), %xmm0
.LBE6984:
.LBE6991:
.LBB6992:
.LBB6988:
	insertps	$0x10, 16(%rbx), %xmm2
.LBE6988:
.LBE6992:
.LBB6993:
.LBB6985:
	movlhps	%xmm1, %xmm0
.LBE6985:
.LBE6993:
.LBB6994:
.LBB6989:
	movss	4(%rbx), %xmm1
	insertps	$0x10, 8(%rbx), %xmm1
.LBE6989:
.LBE6994:
.LBB6995:
.LBB6996:
	.loc 4 189 0 discriminator 2
	mulps	48(%r14), %xmm0
.LBE6996:
.LBE6995:
.LBB6998:
.LBB6990:
	.loc 4 946 0 discriminator 2
	movlhps	%xmm2, %xmm1
.LBE6990:
.LBE6998:
.LBB6999:
.LBB6997:
	.loc 4 189 0 discriminator 2
	mulps	16(%r14), %xmm1
	subps	%xmm1, %xmm0
.LBE6997:
.LBE6999:
.LBB7000:
.LBB7001:
.LBB7002:
.LBB7003:
.LBB7004:
.LBB7005:
	.loc 3 63 0 discriminator 2
	pxor	%xmm1, %xmm1
.LBE7005:
.LBE7004:
.LBE7003:
.LBE7002:
.LBE7001:
.LBE7000:
.LBB7013:
.LBB7014:
	.loc 7 58 0 discriminator 2
	haddps	%xmm0, %xmm0
.LBE7014:
.LBE7013:
.LBB7015:
.LBB7016:
	haddps	%xmm0, %xmm0
.LBE7016:
.LBE7015:
.LBB7017:
.LBB7018:
	.loc 4 960 0 discriminator 2
	movss	%xmm0, (%rbx)
.LBE7018:
.LBE7017:
.LBB7019:
.LBB7012:
.LBB7011:
.LBB7010:
.LBB7007:
.LBB7006:
	.loc 3 63 0 discriminator 2
	movzbl	(%r15), %eax
	cvtsi2ss	%eax, %xmm1
	addss	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
.LBE7006:
.LBE7007:
.LBB7008:
.LBB7009:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm0, %edx
.LBE7009:
.LBE7008:
.LBE7010:
.LBE7011:
.LBE7012:
.LBE7019:
	.loc 1 604 0 discriminator 2
	testl	%edx, %edx
	setg	%al
	negl	%eax
	cmpl	$255, %edx
	cmovbe	%edx, %eax
.LBE6981:
	.loc 1 585 0 discriminator 2
	subq	$4, %r13
	subq	$4, %rbx
.LBB7020:
	.loc 1 604 0 discriminator 2
	movb	%al, (%r15)
.LVL638:
.LBE7020:
	.loc 1 585 0 discriminator 2
	subq	$1, %r15
.LVL639:
	cmpq	%r12, %r13
	jne	.L242
.L241:
	addl	$1, -88(%rbp)
.LVL640:
	addq	$1, %r8
.LBE7023:
	movq	-96(%rbp), %rsp
	movl	-88(%rbp), %eax
.LVL641:
	cmpl	%eax, -152(%rbp)
	jg	.L239
	movl	-168(%rbp), %ebx
.LVL642:
	movl	-192(%rbp), %edi
	movq	%rcx, %r15
.LVL643:
	addl	%ebx, %edi
	movl	%ebx, -88(%rbp)
	movl	%edi, %eax
.LVL644:
	movl	-184(%rbp), %edi
	cmpl	%edi, %eax
	cmovg	%edi, %eax
	movl	%eax, -152(%rbp)
	movl	%ebx, %eax
	movl	-172(%rbp), %ebx
	addl	%ebx, %eax
	movl	%eax, -168(%rbp)
	subl	%ebx, %eax
	cmpl	%eax, %edi
	jg	.L240
.LVL645:
.L236:
	call	GOMP_barrier
.LVL646:
.LBE6875:
.LBE6874:
.LBE6873:
	.loc 1 392 0
	movq	-56(%rbp), %rax
	xorq	%fs:40, %rax
	jne	.L292
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.LVL647:
	.p2align 4,,10
	.p2align 3
.L282:
	.cfi_restore_state
.LBB7027:
.LBB6872:
.LBB6871:
.LBB6682:
.LBB6681:
	.loc 1 405 0
	movl	-172(%rbp), %ecx
.LBE6681:
	.loc 1 404 0
	xorl	%r9d, %r9d
	jmp	.L211
.LVL648:
.L291:
.LBE6682:
.LBE6871:
.LBE6872:
.LBE7027:
.LBB7028:
.LBB7026:
.LBB7025:
.LBB7024:
	.loc 1 556 0
	movq	%r15, %rax
	jmp	.L244
.LVL649:
.L292:
.LBE7024:
.LBE7025:
.LBE7026:
.LBE7028:
	.loc 1 392 0
	call	__stack_chk_fail
.LVL650:
	.cfi_endproc
.LFE12402:
	.size	_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_._omp_fn.3, .-_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_._omp_fn.3
	.section	.text.unlikely._ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_._omp_fn.3,"axG",@progbits,_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_,comdat
.LCOLDE3:
	.section	.text._ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_._omp_fn.3,"axG",@progbits,_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_,comdat
.LHOTE3:
	.section	.text.unlikely._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_._omp_fn.4,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_,comdat
.LCOLDB4:
	.section	.text._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_._omp_fn.4,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_,comdat
.LHOTB4:
	.p2align 4,,15
	.type	_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_._omp_fn.4, @function
_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_._omp_fn.4:
.LFB12403:
	.loc 1 392 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
.LVL651:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movl	$1, %ecx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	leaq	-64(%rbp), %r9
	pushq	%rbx
	leaq	-72(%rbp), %r8
	subq	$216, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movslq	76(%rdi), %rsi
	.loc 1 392 0
	movq	16(%rdi), %r14
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
	movq	24(%rdi), %rax
	movq	%rdi, -88(%rbp)
	movq	%rax, -232(%rbp)
.LVL652:
	movslq	80(%rdi), %rax
.LVL653:
	xorl	%edi, %edi
.LVL654:
	movq	%rax, %rdx
	movl	%eax, -96(%rbp)
.LVL655:
	movq	%rax, %rbx
	movq	%rax, -248(%rbp)
	call	GOMP_loop_dynamic_start
.LVL656:
	testb	%al, %al
	movq	-88(%rbp), %r10
	je	.L294
	movq	%rbx, %rax
	movq	%r14, %r13
	salq	$2, %rax
	movq	%rax, -240(%rbp)
	leaq	96(%r14), %rax
	movq	%r10, %r14
.LVL657:
	movq	%rax, -224(%rbp)
.LVL658:
.L298:
	movq	-72(%rbp), %rax
	movl	-64(%rbp), %edi
	movq	%r14, %r15
	movq	%r13, %r14
	movl	%eax, -172(%rbp)
.LVL659:
	movl	%eax, -176(%rbp)
	cltq
.LVL660:
	movq	%rax, -192(%rbp)
	salq	$2, %rax
	movl	%edi, -216(%rbp)
	movq	%rax, -184(%rbp)
.L297:
.LVL661:
.LBB7029:
.LBB7030:
.LBB7031:
	.loc 1 399 0
	movl	72(%r15), %r9d
	movq	-232(%rbp), %rax
	xorl	%esi, %esi
	addq	-184(%rbp), %rax
	movq	$0, -104(%rbp)
	testl	%r9d, %r9d
	movq	%rax, -120(%rbp)
	jle	.L321
.LVL662:
	.p2align 4,,10
	.p2align 3
.L377:
.LBB7032:
	.loc 1 400 0
	movq	32(%r15), %rax
	movq	-104(%rbp), %rbx
	movq	16(%rax), %rdi
.LVL663:
.LBB7033:
.LBB7034:
	.loc 2 436 0
	movq	72(%rax), %rax
.LVL664:
.LBE7034:
.LBE7033:
	.loc 1 400 0
	movq	%rbx, %r8
	imulq	(%rax), %r8
	movq	-192(%rbp), %rax
	addq	%rax, %r8
	addq	%rdi, %r8
.LVL665:
	.loc 1 401 0
	addq	%rax, %rdi
.LVL666:
	.loc 1 402 0
	movq	48(%r15), %rax
.LVL667:
.LBB7035:
.LBB7036:
	.loc 2 430 0
	movq	72(%rax), %rdx
.LBE7036:
.LBE7035:
	.loc 1 402 0
	imulq	(%rdx), %rbx
	movq	%rbx, %rdx
	addq	-184(%rbp), %rdx
	addq	16(%rax), %rdx
.LVL668:
.LBB7037:
	.loc 1 405 0
	movl	-96(%rbp), %eax
.LVL669:
	testl	%eax, %eax
	jle	.L318
	movl	76(%r15), %eax
	leal	-15(%rax), %ecx
	cmpl	%ecx, -172(%rbp)
	jge	.L397
	leaq	80(%r14), %r13
	leaq	64(%r14), %r12
	movq	%r14, -112(%rbp)
	movq	-120(%rbp), %rbx
	movl	-176(%rbp), %r14d
	xorl	%r9d, %r9d
.LVL670:
	.p2align 4,,10
	.p2align 3
.L336:
	pxor	%xmm1, %xmm1
	movl	$1, %ecx
	movdqa	(%r15), %xmm5
	xorl	%eax, %eax
	movl	%r9d, -88(%rbp)
	movaps	%xmm1, %xmm7
	movaps	%xmm1, %xmm6
	movaps	%xmm1, %xmm8
.LVL671:
.L341:
.LBB7038:
.LBB7039:
.LBB7040:
	.loc 1 409 0
	cmpl	%eax, %esi
	movl	%eax, %r11d
.LVL672:
	jge	.L338
	movq	%rdi, %r10
.L343:
.LVL673:
.LBB7041:
.LBB7042:
	.loc 3 698 0 discriminator 4
	movdqu	(%r10), %xmm2
.LBE7042:
.LBE7041:
.LBB7043:
.LBB7044:
.LBB7045:
.LBB7046:
	.loc 4 884 0 discriminator 4
	movss	0(%r13,%rax,4), %xmm0
.LVL674:
.LBE7046:
.LBE7045:
.LBE7044:
.LBE7043:
	.loc 1 424 0 discriminator 4
	cmpl	%r11d, %esi
.LBB7047:
.LBB7048:
	.loc 3 965 0 discriminator 4
	movdqa	%xmm2, %xmm3
.LBE7048:
.LBE7047:
.LBB7050:
.LBB7051:
	.loc 4 743 0 discriminator 4
	shufps	$0, %xmm0, %xmm0
.LVL675:
.LBE7051:
.LBE7050:
.LBB7052:
.LBB7049:
	.loc 3 965 0 discriminator 4
	punpckhbw	%xmm5, %xmm3
.LVL676:
.LBE7049:
.LBE7052:
.LBB7053:
.LBB7054:
	.loc 3 989 0 discriminator 4
	punpcklbw	%xmm5, %xmm2
.LVL677:
.LBE7054:
.LBE7053:
.LBB7055:
.LBB7056:
	.loc 3 995 0 discriminator 4
	movdqa	%xmm3, %xmm9
.LBE7056:
.LBE7055:
.LBB7058:
.LBB7059:
	.loc 3 971 0 discriminator 4
	punpckhwd	%xmm5, %xmm3
.LVL678:
.LBE7059:
.LBE7058:
.LBB7060:
.LBB7061:
	movdqa	%xmm2, %xmm4
.LBE7061:
.LBE7060:
.LBB7063:
.LBB7057:
	.loc 3 995 0 discriminator 4
	punpcklwd	%xmm5, %xmm9
.LBE7057:
.LBE7063:
.LBB7064:
.LBB7062:
	.loc 3 971 0 discriminator 4
	punpckhwd	%xmm5, %xmm4
.LVL679:
.LBE7062:
.LBE7064:
.LBB7065:
.LBB7066:
	.loc 3 767 0 discriminator 4
	cvtdq2ps	%xmm3, %xmm3
.LBE7066:
.LBE7065:
.LBB7067:
.LBB7068:
	.loc 4 183 0 discriminator 4
	mulps	%xmm0, %xmm3
.LBE7068:
.LBE7067:
.LBB7070:
.LBB7071:
	.loc 3 995 0 discriminator 4
	punpcklwd	%xmm5, %xmm2
.LVL680:
.LBE7071:
.LBE7070:
.LBB7072:
.LBB7073:
	.loc 3 767 0 discriminator 4
	cvtdq2ps	%xmm4, %xmm4
.LVL681:
.LBE7073:
.LBE7072:
.LBB7074:
.LBB7075:
	.loc 4 183 0 discriminator 4
	mulps	%xmm0, %xmm4
.LBE7075:
.LBE7074:
.LBB7077:
.LBB7069:
	addps	%xmm3, %xmm8
.LVL682:
.LBE7069:
.LBE7077:
.LBB7078:
.LBB7079:
	.loc 3 767 0 discriminator 4
	cvtdq2ps	%xmm2, %xmm2
.LVL683:
.LBE7079:
.LBE7078:
.LBB7080:
.LBB7081:
	cvtdq2ps	%xmm9, %xmm3
.LVL684:
.LBE7081:
.LBE7080:
.LBB7082:
.LBB7083:
	.loc 4 183 0 discriminator 4
	mulps	%xmm0, %xmm3
.LBE7083:
.LBE7082:
.LBB7085:
.LBB7086:
	mulps	%xmm2, %xmm0
.LVL685:
.LBE7086:
.LBE7085:
.LBB7088:
.LBB7076:
	addps	%xmm4, %xmm7
.LVL686:
.LBE7076:
.LBE7088:
.LBB7089:
.LBB7084:
	addps	%xmm3, %xmm6
.LVL687:
.LBE7084:
.LBE7089:
.LBB7090:
.LBB7087:
	addps	%xmm0, %xmm1
.LVL688:
.LBE7087:
.LBE7090:
	.loc 1 424 0 discriminator 4
	jg	.L339
	movq	%rbx, %r10
.LVL689:
.L342:
.LBB7091:
.LBB7092:
.LBB7093:
.LBB7094:
	.loc 4 884 0 discriminator 4
	movss	(%r12,%rax,4), %xmm0
.LVL690:
.LBE7094:
.LBE7093:
.LBE7092:
.LBE7091:
.LBB7095:
.LBB7096:
	.loc 4 931 0 discriminator 4
	movups	48(%r10), %xmm9
.LVL691:
	addq	$1, %rax
.LVL692:
	addl	$1, %ecx
.LBE7096:
.LBE7095:
.LBB7097:
.LBB7098:
	.loc 4 743 0 discriminator 4
	shufps	$0, %xmm0, %xmm0
.LVL693:
.LBE7098:
.LBE7097:
.LBB7099:
.LBB7100:
	.loc 4 931 0 discriminator 4
	movups	32(%r10), %xmm4
.LVL694:
.LBE7100:
.LBE7099:
.LBE7040:
	.loc 1 408 0 discriminator 4
	cmpq	$4, %rax
.LBB7121:
.LBB7101:
.LBB7102:
	.loc 4 931 0 discriminator 4
	movups	16(%r10), %xmm3
.LVL695:
.LBE7102:
.LBE7101:
.LBB7103:
.LBB7104:
	.loc 4 189 0 discriminator 4
	mulps	%xmm0, %xmm9
.LVL696:
.LBE7104:
.LBE7103:
.LBB7106:
.LBB7107:
	.loc 4 931 0 discriminator 4
	movups	(%r10), %xmm2
.LVL697:
.LBE7107:
.LBE7106:
.LBB7108:
.LBB7109:
	.loc 4 189 0 discriminator 4
	mulps	%xmm0, %xmm4
.LVL698:
.LBE7109:
.LBE7108:
.LBB7111:
.LBB7112:
	mulps	%xmm0, %xmm3
.LVL699:
.LBE7112:
.LBE7111:
.LBB7114:
.LBB7105:
	subps	%xmm9, %xmm8
.LVL700:
.LBE7105:
.LBE7114:
.LBB7115:
.LBB7116:
	mulps	%xmm2, %xmm0
.LVL701:
.LBE7116:
.LBE7115:
.LBB7118:
.LBB7110:
	subps	%xmm4, %xmm6
.LVL702:
.LBE7110:
.LBE7118:
.LBB7119:
.LBB7113:
	subps	%xmm3, %xmm7
.LVL703:
.LBE7113:
.LBE7119:
.LBB7120:
.LBB7117:
	subps	%xmm0, %xmm1
.LVL704:
.LBE7117:
.LBE7120:
.LBE7121:
	.loc 1 408 0 discriminator 4
	jne	.L341
	movl	-88(%rbp), %r9d
.LVL705:
.LBE7039:
.LBE7038:
	.loc 1 405 0
	addq	$16, %r8
.LVL706:
	addq	$16, %rdi
.LVL707:
.LBB7132:
.LBB7123:
.LBB7124:
	.loc 4 980 0
	movups	%xmm1, (%rdx)
.LVL708:
.LBE7124:
.LBE7123:
.LBE7132:
	.loc 1 405 0
	addq	$64, %rbx
.LVL709:
	addq	$64, %rdx
.LVL710:
.LBB7133:
.LBB7125:
.LBB7126:
	.loc 4 980 0
	movups	%xmm7, -48(%rdx)
.LVL711:
.LBE7126:
.LBE7125:
.LBE7133:
	.loc 1 405 0
	addl	$16, %r9d
.LBB7134:
.LBB7127:
.LBB7128:
	.loc 4 980 0
	movups	%xmm6, -32(%rdx)
.LVL712:
.LBE7128:
.LBE7127:
.LBB7129:
.LBB7130:
	movups	%xmm8, -16(%rdx)
.LVL713:
.LBE7130:
.LBE7129:
.LBE7134:
	.loc 1 405 0
	cmpl	%r9d, -96(%rbp)
	jle	.L398
	.loc 1 405 0 is_stmt 0 discriminator 1
	movl	76(%r15), %eax
	leal	(%r9,%r14), %ecx
	leal	-15(%rax), %r10d
.LVL714:
	cmpl	%ecx, %r10d
	jg	.L336
	movq	-112(%rbp), %r14
.LVL715:
.L319:
.LBE7037:
.LBB7136:
	.loc 1 442 0 is_stmt 1
	cmpl	%ecx, %eax
	jle	.L392
.LBB7137:
	.loc 1 450 0 discriminator 1
	movslq	%eax, %rbx
	.loc 1 445 0 discriminator 1
	leal	(%rax,%rax), %ecx
	movq	%r15, -112(%rbp)
	.loc 1 450 0 discriminator 1
	leaq	0(,%rbx,4), %r10
	.loc 1 445 0 discriminator 1
	movq	%rbx, -88(%rbp)
	movl	-176(%rbp), %r15d
	movslq	%ecx, %r12
	addl	%eax, %ecx
	movl	-96(%rbp), %ebx
	.loc 1 450 0 discriminator 1
	negq	%r10
	.loc 1 445 0 discriminator 1
	movslq	%ecx, %r13
	jmp	.L366
.LVL716:
	.p2align 4,,10
	.p2align 3
.L399:
	.loc 1 445 0 is_stmt 0
	movzbl	(%r8), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2ss	%ecx, %xmm0
	mulss	80(%r14), %xmm0
	addss	(%rdx), %xmm0
	movss	%xmm0, (%rdx)
	.loc 1 449 0 is_stmt 1
	je	.L326
	.loc 1 450 0
	leaq	(%rdx,%r10), %rcx
	movss	64(%r14), %xmm1
	.loc 1 445 0
	movq	%r8, %r11
	subq	-88(%rbp), %r11
	.loc 1 449 0
	cmpl	$1, %esi
	.loc 1 450 0
	mulss	(%rcx), %xmm1
	subss	%xmm1, %xmm0
	.loc 1 445 0
	pxor	%xmm1, %xmm1
	.loc 1 450 0
	movss	%xmm0, (%rdx)
.LVL717:
	.loc 1 445 0
	movzbl	(%r11), %r11d
	cvtsi2ss	%r11d, %xmm1
	mulss	84(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
	.loc 1 449 0
	je	.L328
	.loc 1 450 0
	movss	68(%r14), %xmm1
	addq	%r10, %rcx
	.loc 1 445 0
	movq	%r8, %r11
	.loc 1 450 0
	mulss	(%rcx), %xmm1
	.loc 1 445 0
	subq	%r12, %r11
	.loc 1 449 0
	cmpl	$2, %esi
	.loc 1 450 0
	subss	%xmm1, %xmm0
	.loc 1 445 0
	pxor	%xmm1, %xmm1
	.loc 1 450 0
	movss	%xmm0, (%rdx)
.LVL718:
	.loc 1 445 0
	movzbl	(%r11), %r11d
	cvtsi2ss	%r11d, %xmm1
	mulss	88(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
	.loc 1 449 0
	je	.L330
	.loc 1 450 0
	movss	72(%r14), %xmm1
	.loc 1 445 0
	movq	%r8, %r11
	.loc 1 450 0
	mulss	(%rcx,%r10), %xmm1
	.loc 1 445 0
	subq	%r13, %r11
	.loc 1 449 0
	cmpl	$3, %esi
	.loc 1 450 0
	subss	%xmm1, %xmm0
	.loc 1 445 0
	pxor	%xmm1, %xmm1
	.loc 1 450 0
	movss	%xmm0, (%rdx)
.LVL719:
	.loc 1 445 0
	movzbl	(%r11), %r11d
	cvtsi2ss	%r11d, %xmm1
	mulss	92(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
	.loc 1 449 0
	je	.L332
	.loc 1 450 0
	movss	(%rcx,%r10,2), %xmm1
	mulss	76(%r14), %xmm1
	subss	%xmm1, %xmm0
.L333:
.LBE7137:
	.loc 1 442 0
	addl	$1, %r9d
.LVL720:
	movss	%xmm0, (%rdx)
.LVL721:
	addq	$1, %r8
.LVL722:
	addq	$1, %rdi
.LVL723:
	addq	$4, %rdx
.LVL724:
	cmpl	%r9d, %ebx
	jle	.L393
	.loc 1 442 0 is_stmt 0 discriminator 1
	leal	(%r9,%r15), %ecx
	cmpl	%ecx, %eax
	jle	.L393
.LVL725:
.L366:
.LBB7138:
	.loc 1 444 0 is_stmt 1 discriminator 1
	testl	%esi, %esi
	jns	.L399
	.loc 1 447 0
	movzbl	(%rdi), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2ss	%ecx, %xmm0
	mulss	80(%r14), %xmm0
	addss	(%rdx), %xmm0
	movss	%xmm0, (%rdx)
.L326:
	.loc 1 452 0
	movzbl	(%rdi), %ecx
	pxor	%xmm1, %xmm1
	cvtsi2ss	%ecx, %xmm1
	mulss	64(%r14), %xmm1
	subss	%xmm1, %xmm0
	.loc 1 447 0
	pxor	%xmm1, %xmm1
	.loc 1 452 0
	movss	%xmm0, (%rdx)
.LVL726:
	.loc 1 447 0
	movzbl	(%rdi), %ecx
	cvtsi2ss	%ecx, %xmm1
	mulss	84(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
.L328:
	.loc 1 452 0
	movzbl	(%rdi), %ecx
	pxor	%xmm1, %xmm1
	cvtsi2ss	%ecx, %xmm1
	mulss	68(%r14), %xmm1
	subss	%xmm1, %xmm0
	.loc 1 447 0
	pxor	%xmm1, %xmm1
	.loc 1 452 0
	movss	%xmm0, (%rdx)
.LVL727:
	.loc 1 447 0
	movzbl	(%rdi), %ecx
	cvtsi2ss	%ecx, %xmm1
	mulss	88(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
.L330:
	.loc 1 452 0
	movzbl	(%rdi), %ecx
	pxor	%xmm1, %xmm1
	cvtsi2ss	%ecx, %xmm1
	mulss	72(%r14), %xmm1
	subss	%xmm1, %xmm0
	.loc 1 447 0
	pxor	%xmm1, %xmm1
	.loc 1 452 0
	movss	%xmm0, (%rdx)
.LVL728:
	.loc 1 447 0
	movzbl	(%rdi), %ecx
	cvtsi2ss	%ecx, %xmm1
	mulss	92(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
.L332:
	.loc 1 452 0
	movzbl	(%rdi), %ecx
	pxor	%xmm1, %xmm1
	cvtsi2ss	%ecx, %xmm1
	mulss	76(%r14), %xmm1
	subss	%xmm1, %xmm0
	jmp	.L333
.LVL729:
	.p2align 4,,10
	.p2align 3
.L339:
.LBE7138:
.LBE7136:
.LBB7139:
.LBB7135:
.LBB7131:
.LBB7122:
	.loc 1 424 0 discriminator 1
	movl	76(%r15), %r10d
.LVL730:
	movq	%rdx, %r11
	imull	%ecx, %r10d
	movslq	%r10d, %r10
	salq	$2, %r10
	subq	%r10, %r11
	movq	%r11, %r10
	jmp	.L342
.LVL731:
	.p2align 4,,10
	.p2align 3
.L338:
	.loc 1 409 0 discriminator 1
	movl	76(%r15), %r10d
	movq	%r8, %r9
	imull	%eax, %r10d
	movslq	%r10d, %r10
	subq	%r10, %r9
	movq	%r9, %r10
	jmp	.L343
.LVL732:
	.p2align 4,,10
	.p2align 3
.L398:
	movq	-112(%rbp), %r14
.LVL733:
.L392:
	movl	72(%r15), %r9d
.LVL734:
.L318:
.LBE7122:
.LBE7131:
.LBE7135:
.LBE7139:
.LBE7032:
	.loc 1 399 0 discriminator 2
	addl	$1, %esi
.LVL735:
	addq	$1, -104(%rbp)
	cmpl	%r9d, %esi
	jl	.L377
.LVL736:
.L321:
.LBE7031:
.LBB7142:
	.loc 1 457 0
	leal	-1(%r9), %edx
	movslq	%edx, %rax
	testl	%edx, %edx
	movl	%edx, %ebx
.LVL737:
	movq	%rax, -200(%rbp)
	js	.L302
	movl	-96(%rbp), %r12d
	movl	-176(%rbp), %r10d
	movq	%rax, %r8
.LVL738:
	.p2align 4,,10
	.p2align 3
.L378:
.LBB7143:
	.loc 1 458 0
	movq	32(%r15), %rax
	movq	%r8, %rsi
	.loc 1 459 0
	movslq	%edx, %rdx
	.loc 1 458 0
	movq	-192(%rbp), %rcx
	.loc 1 460 0
	movq	-184(%rbp), %r11
	movq	16(%rax), %rdi
.LVL739:
.LBB7144:
.LBB7145:
	.loc 2 436 0
	movq	72(%rax), %rax
.LVL740:
	movq	(%rax), %rax
.LVL741:
.LBE7145:
.LBE7144:
	.loc 1 459 0
	imulq	%rax, %rdx
	.loc 1 458 0
	imulq	%rax, %rsi
	addq	%rcx, %rsi
	.loc 1 459 0
	addq	%rdx, %rcx
	.loc 1 460 0
	movq	64(%r15), %rdx
	.loc 1 458 0
	addq	%rdi, %rsi
.LVL742:
	.loc 1 459 0
	addq	%rdi, %rcx
.LVL743:
	.loc 1 460 0
	movq	%r8, %rdi
.LBB7146:
.LBB7147:
	.loc 2 430 0
	movq	72(%rdx), %rax
.LBE7147:
.LBE7146:
	.loc 1 460 0
	imulq	(%rax), %rdi
	movq	%rdi, %rax
	.loc 1 461 0
	movq	48(%r15), %rdi
	.loc 1 460 0
	addq	%r11, %rax
	addq	16(%rdx), %rax
.LVL744:
.LBB7148:
.LBB7149:
	.loc 2 430 0
	movq	72(%rdi), %rdx
.LBE7149:
.LBE7148:
	.loc 1 461 0
	imulq	(%rdx), %r8
	movq	%r8, %rdx
	addq	%r11, %rdx
	addq	16(%rdi), %rdx
.LVL745:
.LBB7150:
	.loc 1 463 0
	testl	%r12d, %r12d
	jle	.L299
	movl	76(%r15), %r8d
	leal	-15(%r8), %edi
.LVL746:
	cmpl	%edi, -172(%rbp)
	jge	.L358
	movq	-224(%rbp), %r13
	xorl	%edi, %edi
.LVL747:
	.p2align 4,,10
	.p2align 3
.L301:
	xorl	%r8d, %r8d
	pxor	%xmm0, %xmm0
.LBB7151:
.LBB7152:
.LBB7153:
	.loc 1 467 0
	addl	$1, %r8d
	movq	%r13, %r11
	subl	%r8d, %r9d
	movdqa	(%r15), %xmm4
.LVL748:
	cmpl	%ebx, %r9d
.LBE7153:
.LBE7152:
.LBE7151:
	.loc 1 463 0
	movaps	%xmm0, %xmm7
.LVL749:
	movaps	%xmm0, %xmm6
.LVL750:
	movaps	%xmm0, %xmm3
.LVL751:
.LBB7319:
.LBB7286:
.LBB7282:
	.loc 1 467 0
	jg	.L316
.LVL752:
.L400:
.LBB7154:
	.loc 1 496 0
	movss	(%r11), %xmm1
	addq	$4, %r11
.LBB7155:
.LBB7156:
	.loc 3 698 0
	movdqu	(%rcx), %xmm2
.LBE7156:
.LBE7155:
	.loc 1 496 0
	subss	-36(%r11), %xmm1
.LVL753:
.LBE7154:
.LBE7282:
	.loc 1 466 0
	cmpl	$4, %r8d
.LBB7283:
.LBB7197:
.LBB7157:
.LBB7158:
	.loc 3 965 0
	movdqa	%xmm2, %xmm5
.LBE7158:
.LBE7157:
.LBB7160:
.LBB7161:
	.loc 3 989 0
	punpcklbw	%xmm4, %xmm2
.LBE7161:
.LBE7160:
.LBB7162:
.LBB7159:
	.loc 3 965 0
	punpckhbw	%xmm4, %xmm5
.LBE7159:
.LBE7162:
.LBB7163:
.LBB7164:
	.loc 3 971 0
	movdqa	%xmm2, %xmm8
.LBE7164:
.LBE7163:
.LBB7166:
.LBB7167:
	.loc 3 995 0
	punpcklwd	%xmm4, %xmm2
.LBE7167:
.LBE7166:
.LBB7168:
.LBB7169:
	.loc 4 891 0
	shufps	$0, %xmm1, %xmm1
.LVL754:
.LBE7169:
.LBE7168:
.LBB7170:
.LBB7171:
	.loc 3 995 0
	movdqa	%xmm5, %xmm9
.LBE7171:
.LBE7170:
.LBB7173:
.LBB7174:
	.loc 3 971 0
	punpckhwd	%xmm4, %xmm5
.LVL755:
.LBE7174:
.LBE7173:
.LBB7175:
.LBB7176:
	.loc 3 767 0
	cvtdq2ps	%xmm2, %xmm2
.LBE7176:
.LBE7175:
.LBB7177:
.LBB7165:
	.loc 3 971 0
	punpckhwd	%xmm4, %xmm8
.LVL756:
.LBE7165:
.LBE7177:
.LBB7178:
.LBB7172:
	.loc 3 995 0
	punpcklwd	%xmm4, %xmm9
.LVL757:
.LBE7172:
.LBE7178:
.LBB7179:
.LBB7180:
	.loc 3 767 0
	cvtdq2ps	%xmm5, %xmm5
.LVL758:
.LBE7180:
.LBE7179:
.LBB7181:
.LBB7182:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
	addps	%xmm5, %xmm3
.LVL759:
.LBE7182:
.LBE7181:
.LBB7183:
.LBB7184:
	.loc 3 767 0
	cvtdq2ps	%xmm9, %xmm5
.LVL760:
.LBE7184:
.LBE7183:
.LBB7185:
.LBB7186:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
	addps	%xmm5, %xmm6
.LVL761:
.LBE7186:
.LBE7185:
.LBB7187:
.LBB7188:
	.loc 3 767 0
	cvtdq2ps	%xmm8, %xmm5
.LVL762:
.LBE7188:
.LBE7187:
.LBB7189:
.LBB7190:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
.LBE7190:
.LBE7189:
.LBB7192:
.LBB7193:
	mulps	%xmm2, %xmm1
.LBE7193:
.LBE7192:
.LBB7195:
.LBB7191:
	addps	%xmm5, %xmm7
.LVL763:
.LBE7191:
.LBE7195:
.LBB7196:
.LBB7194:
	addps	%xmm1, %xmm0
.LVL764:
.LBE7194:
.LBE7196:
.LBE7197:
.LBE7283:
	.loc 1 466 0
	je	.L313
.LVL765:
.L401:
	movl	72(%r15), %r9d
.LBB7284:
	.loc 1 467 0
	addl	$1, %r8d
.LVL766:
	subl	%r8d, %r9d
	cmpl	%ebx, %r9d
	jle	.L400
.LVL767:
.L316:
.LBB7198:
.LBB7199:
.LBB7200:
	.loc 3 698 0
	movl	76(%r15), %r9d
.LBE7200:
.LBE7199:
.LBB7202:
.LBB7203:
.LBB7204:
.LBB7205:
	.loc 4 884 0
	movss	(%r11), %xmm1
.LVL768:
	addq	$4, %r11
.LBE7205:
.LBE7204:
.LBE7203:
.LBE7202:
.LBB7206:
.LBB7207:
	.loc 4 743 0
	shufps	$0, %xmm1, %xmm1
.LVL769:
.LBE7207:
.LBE7206:
.LBB7208:
.LBB7201:
	.loc 3 698 0
	imull	%r8d, %r9d
	movslq	%r9d, %r9
	movdqu	(%rsi,%r9), %xmm2
.LVL770:
.LBE7201:
.LBE7208:
.LBB7209:
.LBB7210:
	.loc 3 965 0
	movdqa	%xmm2, %xmm5
.LBE7210:
.LBE7209:
.LBB7212:
.LBB7213:
	.loc 3 989 0
	punpcklbw	%xmm4, %xmm2
.LVL771:
.LBE7213:
.LBE7212:
.LBB7214:
.LBB7211:
	.loc 3 965 0
	punpckhbw	%xmm4, %xmm5
.LVL772:
.LBE7211:
.LBE7214:
.LBB7215:
.LBB7216:
	.loc 3 971 0
	movdqa	%xmm2, %xmm8
.LBE7216:
.LBE7215:
.LBB7218:
.LBB7219:
	.loc 3 995 0
	movdqa	%xmm5, %xmm9
.LBE7219:
.LBE7218:
.LBB7221:
.LBB7222:
	.loc 3 971 0
	punpckhwd	%xmm4, %xmm5
.LVL773:
.LBE7222:
.LBE7221:
.LBB7223:
.LBB7217:
	punpckhwd	%xmm4, %xmm8
.LVL774:
.LBE7217:
.LBE7223:
.LBB7224:
.LBB7220:
	.loc 3 995 0
	punpcklwd	%xmm4, %xmm9
.LBE7220:
.LBE7224:
.LBB7225:
.LBB7226:
	.loc 3 767 0
	cvtdq2ps	%xmm5, %xmm5
.LBE7226:
.LBE7225:
.LBB7227:
.LBB7228:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
.LBE7228:
.LBE7227:
.LBB7230:
.LBB7231:
	.loc 3 995 0
	punpcklwd	%xmm4, %xmm2
.LVL775:
.LBE7231:
.LBE7230:
.LBB7232:
.LBB7229:
	.loc 4 183 0
	addps	%xmm5, %xmm3
.LVL776:
.LBE7229:
.LBE7232:
.LBB7233:
.LBB7234:
	.loc 3 767 0
	cvtdq2ps	%xmm9, %xmm5
.LVL777:
.LBE7234:
.LBE7233:
.LBB7235:
.LBB7236:
	cvtdq2ps	%xmm2, %xmm2
.LVL778:
.LBE7236:
.LBE7235:
.LBB7237:
.LBB7238:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
.LBE7238:
.LBE7237:
.LBB7240:
.LBB7241:
	.loc 4 931 0
	movups	48(%rax,%r9,4), %xmm9
.LVL779:
.LBE7241:
.LBE7240:
.LBB7242:
.LBB7239:
	.loc 4 183 0
	addps	%xmm5, %xmm6
.LVL780:
.LBE7239:
.LBE7242:
.LBB7243:
.LBB7244:
	.loc 3 767 0
	cvtdq2ps	%xmm8, %xmm5
.LVL781:
.LBE7244:
.LBE7243:
.LBB7245:
.LBB7246:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
.LBE7246:
.LBE7245:
.LBB7248:
.LBB7249:
	.loc 4 931 0
	movups	32(%rax,%r9,4), %xmm8
.LVL782:
.LBE7249:
.LBE7248:
.LBB7250:
.LBB7251:
	.loc 4 183 0
	mulps	%xmm2, %xmm1
.LVL783:
.LBE7251:
.LBE7250:
.LBB7253:
.LBB7254:
	.loc 4 931 0
	movups	(%rax,%r9,4), %xmm2
.LBE7254:
.LBE7253:
.LBB7255:
.LBB7247:
	.loc 4 183 0
	addps	%xmm5, %xmm7
.LVL784:
.LBE7247:
.LBE7255:
.LBB7256:
.LBB7257:
	.loc 4 931 0
	movups	16(%rax,%r9,4), %xmm5
.LBE7257:
.LBE7256:
.LBB7258:
.LBB7252:
	.loc 4 183 0
	addps	%xmm1, %xmm0
.LVL785:
.LBE7252:
.LBE7258:
.LBB7259:
.LBB7260:
.LBB7261:
.LBB7262:
	.loc 4 884 0
	movss	-36(%r11), %xmm1
.LVL786:
.LBE7262:
.LBE7261:
.LBE7260:
.LBE7259:
.LBE7198:
.LBE7284:
	.loc 1 466 0
	cmpl	$4, %r8d
.LBB7285:
.LBB7281:
.LBB7263:
.LBB7264:
	.loc 4 743 0
	shufps	$0, %xmm1, %xmm1
.LVL787:
.LBE7264:
.LBE7263:
.LBB7265:
.LBB7266:
	.loc 4 189 0
	mulps	%xmm1, %xmm9
.LVL788:
.LBE7266:
.LBE7265:
.LBB7268:
.LBB7269:
	mulps	%xmm1, %xmm8
.LVL789:
.LBE7269:
.LBE7268:
.LBB7271:
.LBB7272:
	mulps	%xmm1, %xmm5
.LVL790:
.LBE7272:
.LBE7271:
.LBB7274:
.LBB7267:
	subps	%xmm9, %xmm3
.LBE7267:
.LBE7274:
.LBB7275:
.LBB7276:
	mulps	%xmm2, %xmm1
.LVL791:
.LBE7276:
.LBE7275:
.LBB7278:
.LBB7270:
	subps	%xmm8, %xmm6
.LBE7270:
.LBE7278:
.LBB7279:
.LBB7273:
	subps	%xmm5, %xmm7
.LBE7273:
.LBE7279:
.LBB7280:
.LBB7277:
	subps	%xmm1, %xmm0
.LVL792:
.LBE7277:
.LBE7280:
.LBE7281:
.LBE7285:
	.loc 1 466 0
	jne	.L401
.LVL793:
.L313:
.LBE7286:
.LBB7287:
.LBB7288:
	.loc 4 980 0
	movups	%xmm0, (%rax)
.LVL794:
.LBE7288:
.LBE7287:
.LBE7319:
	.loc 1 463 0
	addl	$16, %edi
.LVL795:
	addq	$16, %rsi
.LVL796:
	addq	$16, %rcx
.LVL797:
.LBB7320:
.LBB7289:
.LBB7290:
	.loc 4 980 0
	movups	%xmm7, 16(%rax)
.LVL798:
.LBE7290:
.LBE7289:
.LBE7320:
	.loc 1 463 0
	addq	$64, %rdx
.LVL799:
	addq	$64, %rax
.LVL800:
.LBB7321:
.LBB7291:
.LBB7292:
	.loc 4 980 0
	movups	%xmm6, -32(%rax)
.LVL801:
.LBE7292:
.LBE7291:
.LBB7293:
.LBB7294:
	movups	%xmm3, -16(%rax)
.LVL802:
.LBE7294:
.LBE7293:
.LBB7295:
.LBB7296:
	.loc 4 931 0
	movups	-16(%rdx), %xmm1
.LVL803:
.LBE7296:
.LBE7295:
.LBB7297:
.LBB7298:
	movups	-32(%rdx), %xmm2
.LVL804:
.LBE7298:
.LBE7297:
.LBB7299:
.LBB7300:
	.loc 4 980 0
	addps	%xmm1, %xmm3
.LBE7300:
.LBE7299:
.LBB7302:
.LBB7303:
	.loc 4 931 0
	movups	-48(%rdx), %xmm4
.LVL805:
.LBE7303:
.LBE7302:
.LBB7304:
.LBB7305:
	.loc 4 980 0
	addps	%xmm2, %xmm6
.LBE7305:
.LBE7304:
.LBB7307:
.LBB7308:
	.loc 4 931 0
	movups	-64(%rdx), %xmm5
.LVL806:
.LBE7308:
.LBE7307:
.LBB7309:
.LBB7310:
	.loc 4 980 0
	addps	%xmm4, %xmm7
.LBE7310:
.LBE7309:
.LBB7312:
.LBB7301:
	movups	%xmm3, -16(%rdx)
.LBE7301:
.LBE7312:
.LBB7313:
.LBB7314:
	addps	%xmm5, %xmm0
.LBE7314:
.LBE7313:
.LBB7316:
.LBB7306:
	movups	%xmm6, -32(%rdx)
.LBE7306:
.LBE7316:
.LBB7317:
.LBB7311:
	movups	%xmm7, -48(%rdx)
.LBE7311:
.LBE7317:
.LBB7318:
.LBB7315:
	movups	%xmm0, -64(%rdx)
.LVL807:
.LBE7315:
.LBE7318:
.LBE7321:
	.loc 1 463 0
	cmpl	%edi, %r12d
	jle	.L299
	leal	(%rdi,%r10), %r9d
	.loc 1 463 0 is_stmt 0 discriminator 1
	movl	76(%r15), %r8d
.LVL808:
	leal	-15(%r8), %r11d
	cmpl	%r9d, %r11d
	jle	.L300
	movl	72(%r15), %r9d
	jmp	.L301
.LVL809:
	.p2align 4,,10
	.p2align 3
.L387:
	movq	-208(%rbp), %r15
.LVL810:
.L299:
.LBE7150:
.LBE7143:
	.loc 1 457 0 is_stmt 1
	subl	$1, %ebx
.LVL811:
	subq	$1, -200(%rbp)
	cmpl	$-1, %ebx
	je	.L302
	movl	72(%r15), %r9d
	movq	-200(%rbp), %r8
	leal	-1(%r9), %edx
.LVL812:
	jmp	.L378
.LVL813:
.L358:
.LBB7329:
.LBB7322:
	.loc 1 463 0
	movl	-172(%rbp), %r9d
.LBE7322:
	.loc 1 462 0
	xorl	%edi, %edi
.LVL814:
	.p2align 4,,10
	.p2align 3
.L300:
.LBB7323:
	.loc 1 525 0
	cmpl	%r8d, %r9d
	jge	.L299
	movl	72(%r15), %r9d
	movq	%r15, -208(%rbp)
	movl	%r8d, -112(%rbp)
	leal	-1(%r9), %r11d
	leal	-2(%r9), %r13d
	movl	%r11d, -88(%rbp)
	leal	-3(%r9), %r11d
	movl	%r11d, -104(%rbp)
	leal	-4(%r9), %r11d
	movl	%r11d, -212(%rbp)
.LBB7324:
.LBB7325:
	.loc 1 528 0
	movslq	%r8d, %r11
	movl	-212(%rbp), %r15d
	leaq	0(,%r11,4), %r9
	movq	%r11, -120(%rbp)
	movq	%r9, -128(%rbp)
	leal	(%r8,%r8), %r9d
	movslq	%r9d, %r11
	addl	%r8d, %r9d
	movq	%r11, -136(%rbp)
	salq	$2, %r11
	movq	%r11, -144(%rbp)
	movslq	%r9d, %r11
	leaq	0(,%r11,4), %r9
	movq	%r11, -152(%rbp)
	leal	0(,%r8,4), %r11d
	movslq	%r11d, %r11
	movq	%r9, -160(%rbp)
	leaq	0(,%r11,4), %r9
	addq	%r11, %rsi
.LVL815:
	movq	%r9, -168(%rbp)
	jmp	.L355
.LVL816:
	.p2align 4,,10
	.p2align 3
.L402:
	movq	-120(%rbp), %r8
	movq	%rsi, %r9
	subq	%r11, %r9
	pxor	%xmm2, %xmm2
	movss	64(%r14), %xmm0
	.loc 1 527 0
	cmpl	%r13d, %ebx
	.loc 1 528 0
	movzbl	(%r9,%r8), %r9d
	movq	-128(%rbp), %r8
	movss	(%rax), %xmm3
	mulss	(%rax,%r8), %xmm0
	cvtsi2ss	%r9d, %xmm2
	mulss	96(%r14), %xmm2
	subss	%xmm0, %xmm2
	addss	%xmm2, %xmm3
	movss	%xmm3, (%rax)
.LVL817:
	.loc 1 527 0
	jge	.L306
.L403:
	.loc 1 528 0
	movq	-136(%rbp), %r8
	movq	%rsi, %r9
	subq	%r11, %r9
	pxor	%xmm1, %xmm1
	movss	68(%r14), %xmm0
	.loc 1 527 0
	cmpl	-104(%rbp), %ebx
	.loc 1 528 0
	movzbl	(%r9,%r8), %r9d
	movq	-144(%rbp), %r8
	mulss	(%rax,%r8), %xmm0
	cvtsi2ss	%r9d, %xmm1
	mulss	100(%r14), %xmm1
	subss	%xmm0, %xmm1
	movaps	%xmm1, %xmm2
	addss	%xmm3, %xmm2
	movss	%xmm2, (%rax)
.LVL818:
	.loc 1 527 0
	jge	.L308
.L404:
	.loc 1 528 0
	movq	-152(%rbp), %r8
	movq	%rsi, %r9
	subq	%r11, %r9
	pxor	%xmm0, %xmm0
	movss	72(%r14), %xmm1
	.loc 1 527 0
	cmpl	%r15d, %ebx
	.loc 1 528 0
	movzbl	(%r9,%r8), %r9d
	movq	-160(%rbp), %r8
	mulss	(%rax,%r8), %xmm1
	cvtsi2ss	%r9d, %xmm0
	mulss	104(%r14), %xmm0
	subss	%xmm1, %xmm0
	movaps	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	movss	%xmm1, (%rax)
.LVL819:
	.loc 1 527 0
	jge	.L310
.L405:
	.loc 1 528 0
	movzbl	(%rsi), %r9d
	pxor	%xmm0, %xmm0
	movq	-168(%rbp), %r8
	movss	76(%r14), %xmm2
	cvtsi2ss	%r9d, %xmm0
	mulss	(%rax,%r8), %xmm2
	mulss	108(%r14), %xmm0
	subss	%xmm2, %xmm0
	addss	%xmm1, %xmm0
.L311:
	movss	%xmm0, (%rax)
.LVL820:
.LBE7325:
.LBE7324:
	.loc 1 525 0
	addl	$1, %edi
.LVL821:
	addq	$1, %rcx
.LVL822:
	addq	$4, %rax
.LVL823:
	addq	$4, %rdx
.LVL824:
.LBB7327:
	.loc 1 533 0
	addss	-4(%rdx), %xmm0
	movss	%xmm0, -4(%rdx)
.LVL825:
.LBE7327:
	.loc 1 525 0
	cmpl	%edi, %r12d
	jle	.L387
	addq	$1, %rsi
.LVL826:
	.loc 1 525 0 is_stmt 0 discriminator 1
	leal	(%rdi,%r10), %r9d
	cmpl	-112(%rbp), %r9d
	jge	.L387
.LVL827:
.L355:
.LBB7328:
.LBB7326:
	.loc 1 527 0 is_stmt 1 discriminator 1
	cmpl	-88(%rbp), %ebx
	jl	.L402
	.loc 1 530 0
	movzbl	(%rcx), %r9d
	pxor	%xmm0, %xmm0
	movss	96(%r14), %xmm2
	.loc 1 527 0
	cmpl	%r13d, %ebx
	.loc 1 530 0
	subss	64(%r14), %xmm2
	movss	(%rax), %xmm3
	cvtsi2ss	%r9d, %xmm0
	mulss	%xmm0, %xmm2
	addss	%xmm2, %xmm3
	movss	%xmm3, (%rax)
.LVL828:
	.loc 1 527 0
	jl	.L403
.L306:
	.loc 1 530 0
	movzbl	(%rcx), %r9d
	pxor	%xmm0, %xmm0
	movss	100(%r14), %xmm1
	subss	68(%r14), %xmm1
	.loc 1 527 0
	cmpl	-104(%rbp), %ebx
	.loc 1 530 0
	cvtsi2ss	%r9d, %xmm0
	mulss	%xmm0, %xmm1
	movaps	%xmm1, %xmm2
	addss	%xmm3, %xmm2
	movss	%xmm2, (%rax)
.LVL829:
	.loc 1 527 0
	jl	.L404
.L308:
	.loc 1 530 0
	movzbl	(%rcx), %r9d
	pxor	%xmm3, %xmm3
	movss	104(%r14), %xmm0
	.loc 1 527 0
	cmpl	%r15d, %ebx
	.loc 1 530 0
	subss	72(%r14), %xmm0
	cvtsi2ss	%r9d, %xmm3
	mulss	%xmm3, %xmm0
	movaps	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	movss	%xmm1, (%rax)
.LVL830:
	.loc 1 527 0
	jl	.L405
.L310:
	.loc 1 530 0
	movzbl	(%rcx), %r9d
	pxor	%xmm2, %xmm2
	movss	108(%r14), %xmm0
	subss	76(%r14), %xmm0
	cvtsi2ss	%r9d, %xmm2
	mulss	%xmm2, %xmm0
	addss	%xmm1, %xmm0
	jmp	.L311
.LVL831:
	.p2align 4,,10
	.p2align 3
.L393:
	movq	-112(%rbp), %r15
	movl	72(%r15), %r9d
.LVL832:
	jmp	.L318
.LVL833:
.L302:
	movl	-96(%rbp), %edi
	addl	%edi, -172(%rbp)
	movq	-248(%rbp), %rbx
.LVL834:
	movl	-172(%rbp), %eax
.LVL835:
	addq	%rbx, -192(%rbp)
	addl	%edi, -176(%rbp)
	movq	-240(%rbp), %rbx
	addq	%rbx, -184(%rbp)
	cmpl	%eax, -216(%rbp)
	jg	.L297
	leaq	-64(%rbp), %rsi
	leaq	-72(%rbp), %rdi
	movq	%r14, %r13
	movq	%r15, %r14
	call	GOMP_loop_dynamic_next
.LVL836:
	testb	%al, %al
	jne	.L298
	movq	%r15, %r10
	movq	%r13, %r14
.LVL837:
.L294:
	movq	%r10, -88(%rbp)
	call	GOMP_loop_end
.LVL838:
.LBE7326:
.LBE7328:
.LBE7323:
.LBE7329:
.LBE7142:
.LBE7030:
.LBE7029:
.LBB7332:
	.loc 1 539 0
	movq	-88(%rbp), %r10
	movl	72(%r10), %eax
	movq	%r10, -96(%rbp)
	movl	%eax, %r15d
	movl	%eax, -172(%rbp)
	addl	$3, %eax
	testl	%r15d, %r15d
	cmovns	%r15d, %eax
	sarl	$2, %eax
	movl	%eax, %ebx
	call	omp_get_num_threads
.LVL839:
	movl	%eax, %r12d
	call	omp_get_thread_num
.LVL840:
	movl	%ebx, %edi
	imull	%eax, %edi
	leal	(%rdi,%rbx), %edx
	movl	%edi, %esi
	movl	%edi, -88(%rbp)
	cmpl	%r15d, %edx
	cmovg	%r15d, %edx
	cmpl	%r15d, %esi
	movl	%edx, -152(%rbp)
	jge	.L344
	movslq	%eax, %rdi
	addl	%r12d, %eax
	movq	-96(%rbp), %r10
	movq	%rdi, -160(%rbp)
	movl	%ebx, %edi
	imull	%r12d, %edi
	movq	%r10, %r15
	movl	%edi, -184(%rbp)
	movl	%eax, %edi
	addl	$1, %eax
	imull	%ebx, %edi
	imull	%ebx, %eax
	movl	%edi, -168(%rbp)
	subl	%edi, %eax
	movl	%eax, -192(%rbp)
.L348:
	movslq	-88(%rbp), %r8
.L347:
.LVL841:
.LBB7333:
.LBB7334:
	.loc 1 543 0
	movl	76(%r15), %eax
	.loc 1 605 0
	movq	%rsp, -96(%rbp)
.LVL842:
.LBB7335:
.LBB7336:
	.loc 2 430 0
	movq	%r8, %rdi
.LBE7336:
.LBE7335:
.LBB7341:
.LBB7342:
	movq	-160(%rbp), %r10
.LBE7342:
.LBE7341:
.LBB7344:
.LBB7345:
	movq	%r8, %r13
.LBE7345:
.LBE7344:
.LBB7350:
.LBB7351:
	.loc 5 90 0
	xorl	%esi, %esi
.LBE7351:
.LBE7350:
.LBB7354:
.LBB7346:
	.loc 2 430 0
	movq	%r8, -144(%rbp)
.LBE7346:
.LBE7354:
	.loc 1 543 0
	addl	$16, %eax
	cltq
	leaq	18(,%rax,4), %rax
	andq	$-16, %rax
	subq	%rax, %rsp
	.loc 1 545 0
	movq	48(%r15), %rax
	.loc 1 543 0
	leaq	3(%rsp), %r9
.LBB7355:
.LBB7337:
	.loc 2 430 0
	movq	72(%rax), %rdx
.LBE7337:
.LBE7355:
	.loc 1 543 0
	shrq	$2, %r9
	leaq	0(,%r9,4), %r12
.LVL843:
	movq	%r9, -136(%rbp)
.LBB7356:
.LBB7338:
	.loc 2 430 0
	imulq	(%rdx), %rdi
.LBE7338:
.LBE7356:
	.loc 1 552 0
	addq	$32, %r12
.LVL844:
.LBB7357:
.LBB7339:
	.loc 2 430 0
	movq	%rdi, %rdx
	addq	16(%rax), %rdx
.LBE7339:
.LBE7357:
	.loc 1 546 0
	movq	56(%r15), %rax
.LVL845:
.LBB7358:
.LBB7340:
	.loc 2 430 0
	movq	%rdx, -104(%rbp)
.LBE7340:
.LBE7358:
.LBB7359:
.LBB7343:
	movq	72(%rax), %rdx
	imulq	(%rdx), %r10
	addq	16(%rax), %r10
.LVL846:
.LBE7343:
.LBE7359:
	.loc 1 547 0
	movq	40(%r15), %rax
.LVL847:
.LBB7360:
.LBB7347:
	.loc 2 430 0
	movq	72(%rax), %rdx
.LBE7347:
.LBE7360:
	.loc 1 551 0
	leaq	32(%r10), %rbx
	movq	%r10, -128(%rbp)
.LBB7361:
.LBB7348:
	.loc 2 430 0
	imulq	(%rdx), %r13
.LBE7348:
.LBE7361:
.LBB7362:
.LBB7352:
	.loc 5 90 0
	movq	%rbx, %rdi
.LBE7352:
.LBE7362:
.LBB7363:
.LBB7349:
	.loc 2 430 0
	addq	16(%rax), %r13
.LVL848:
	movq	0(,%r9,4), %rax
.LVL849:
.LBE7349:
.LBE7363:
.LBB7364:
.LBB7365:
	.loc 5 53 0
	movq	%rax, 8(%r10)
.LVL850:
.LBE7365:
.LBE7364:
.LBB7366:
.LBB7367:
	movq	%rax, (%r10)
.LVL851:
.LBE7367:
.LBE7366:
.LBB7368:
.LBB7369:
	movq	8(%r10), %rdx
	movq	(%r10), %rax
	movq	%rdx, 24(%r10)
.LVL852:
	movq	%rax, 16(%r10)
.LVL853:
.LBE7369:
.LBE7368:
.LBB7370:
.LBB7353:
	.loc 5 90 0
	movslq	76(%r15), %rdx
	salq	$2, %rdx
	call	memset
.LVL854:
.LBE7353:
.LBE7370:
	.loc 1 552 0
	movslq	76(%r15), %rcx
.LBB7371:
.LBB7372:
	.loc 5 53 0
	movq	-104(%rbp), %rsi
	movq	%r12, %rdi
	leaq	0(,%rcx,4), %rdx
.LBE7372:
.LBE7371:
	.loc 1 552 0
	movl	%ecx, -120(%rbp)
.LVL855:
.LBB7374:
.LBB7373:
	.loc 5 53 0
	movq	%rcx, -112(%rbp)
	call	memcpy
.LVL856:
.LBE7373:
.LBE7374:
.LBB7375:
.LBB7376:
	movq	-128(%rbp), %r10
	movq	-136(%rbp), %r9
.LBE7376:
.LBE7375:
.LBB7379:
	.loc 1 556 0
	movl	-120(%rbp), %r11d
	movq	-112(%rbp), %rcx
	movq	-144(%rbp), %r8
.LBE7379:
.LBB7412:
.LBB7377:
	.loc 5 53 0
	movq	(%r10), %rax
.LBE7377:
.LBE7412:
.LBB7413:
	.loc 1 556 0
	testl	%r11d, %r11d
.LBE7413:
.LBB7414:
.LBB7378:
	.loc 5 53 0
	movq	%rax, 0(,%r9,4)
	movq	8(%r10), %rax
	movq	%rax, 8(,%r9,4)
	movq	16(%r10), %rax
	movq	%rax, 16(,%r9,4)
	movq	24(%r10), %rax
	movq	%rax, 24(,%r9,4)
.LVL857:
.LBE7378:
.LBE7414:
.LBB7415:
	.loc 1 556 0
	jle	.L406
	movss	24(,%r9,4), %xmm1
	.loc 1 556 0 is_stmt 0 discriminator 2
	xorl	%esi, %esi
	movss	28(,%r9,4), %xmm5
	movss	8(,%r9,4), %xmm0
	movss	12(,%r9,4), %xmm4
.LVL858:
	.p2align 4,,10
	.p2align 3
.L353:
.LBB7380:
	.loc 1 567 0 is_stmt 1 discriminator 2
	movss	-16(%r12), %xmm2
.LBB7381:
.LBB7382:
	.loc 4 946 0 discriminator 2
	movaps	%xmm2, %xmm6
.LBE7382:
.LBE7381:
	.loc 1 567 0 discriminator 2
	movss	(%r12), %xmm3
.LBB7385:
.LBB7383:
	.loc 4 946 0 discriminator 2
	unpcklps	%xmm0, %xmm6
	movaps	%xmm3, %xmm0
	unpcklps	%xmm1, %xmm0
.LBE7383:
.LBE7385:
.LBB7386:
.LBB7387:
	movss	-8(%rbx), %xmm1
	insertps	$0x10, -16(%rbx), %xmm1
.LBE7387:
.LBE7386:
.LBB7390:
.LBB7384:
	movlhps	%xmm6, %xmm0
.LBE7384:
.LBE7390:
.LBB7391:
.LBB7388:
	movss	-24(%rbx), %xmm6
	insertps	$0x10, -32(%rbx), %xmm6
.LBE7388:
.LBE7391:
.LBB7392:
.LBB7393:
	.loc 4 189 0 discriminator 2
	mulps	32(%r14), %xmm0
.LBE7393:
.LBE7392:
.LBB7395:
.LBB7389:
	.loc 4 946 0 discriminator 2
	movlhps	%xmm6, %xmm1
.LBE7389:
.LBE7395:
.LBB7396:
.LBB7394:
	.loc 4 189 0 discriminator 2
	mulps	16(%r14), %xmm1
	subps	%xmm1, %xmm0
	movaps	%xmm5, %xmm1
	movaps	%xmm3, %xmm5
.LBE7394:
.LBE7396:
.LBB7397:
.LBB7398:
	.loc 7 58 0 discriminator 2
	haddps	%xmm0, %xmm0
.LBE7398:
.LBE7397:
.LBB7399:
.LBB7400:
	haddps	%xmm0, %xmm0
.LBE7400:
.LBE7399:
.LBB7401:
.LBB7402:
	.loc 4 960 0 discriminator 2
	movss	%xmm0, (%rbx)
.LBE7402:
.LBE7401:
.LBB7403:
.LBB7404:
.LBB7405:
.LBB7406:
.LBB7407:
.LBB7408:
	.loc 3 63 0 discriminator 2
	cvtss2sd	%xmm0, %xmm0
.LBE7408:
.LBE7407:
.LBB7409:
.LBB7410:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm0, %edx
	movaps	%xmm4, %xmm0
	movaps	%xmm2, %xmm4
.LBE7410:
.LBE7409:
.LBE7406:
.LBE7405:
.LBE7404:
.LBE7403:
	.loc 1 575 0 discriminator 2
	testl	%edx, %edx
	setg	%al
	negl	%eax
	cmpl	$255, %edx
	cmovbe	%edx, %eax
.LBE7380:
	.loc 1 556 0 discriminator 2
	addl	$1, %esi
.LVL859:
	addq	$4, %r12
.LVL860:
.LBB7411:
	.loc 1 575 0 discriminator 2
	movb	%al, 0(%r13)
.LBE7411:
	.loc 1 556 0 discriminator 2
	movslq	76(%r15), %rax
	addq	$4, %rbx
.LVL861:
	addq	$1, %r13
.LVL862:
	cmpl	%esi, %eax
	jg	.L353
.LVL863:
.L354:
.LBE7415:
	.loc 1 577 0
	movq	40(%r15), %rdx
.LVL864:
	movq	%r8, %rdi
	movq	%r8, -104(%rbp)
.LBB7416:
.LBB7417:
	.loc 2 430 0
	movq	72(%rdx), %rsi
.LBE7417:
.LBE7416:
	.loc 1 577 0
	imulq	(%rsi), %rdi
.LBB7418:
.LBB7419:
	.loc 5 90 0
	xorl	%esi, %esi
.LBE7419:
.LBE7418:
	.loc 1 577 0
	leaq	-1(%rax,%rdi), %r13
.LVL865:
	movq	-8(%r12), %rax
	addq	16(%rdx), %r13
.LVL866:
.LBB7422:
.LBB7420:
	.loc 5 90 0
	movq	%rbx, %rdi
.LBE7420:
.LBE7422:
	.loc 1 584 0
	subq	$4, %rbx
.LVL867:
.LBB7423:
.LBB7424:
	.loc 5 53 0
	movq	%rax, 12(%rbx)
.LBE7424:
.LBE7423:
.LBB7425:
.LBB7426:
	movq	%rax, 4(%rbx)
.LVL868:
.LBE7426:
.LBE7425:
.LBB7427:
.LBB7428:
	movq	12(%rbx), %rdx
.LVL869:
	movq	4(%rbx), %rax
	movq	%rdx, 28(%rbx)
.LVL870:
	movq	%rax, 20(%rbx)
.LBE7428:
.LBE7427:
	.loc 1 581 0
	movslq	76(%r15), %rdx
	salq	$2, %rdx
.LVL871:
.LBB7429:
.LBB7421:
	.loc 5 90 0
	subq	%rdx, %rdi
.LVL872:
	call	memset
.LVL873:
.LBE7421:
.LBE7429:
.LBB7430:
.LBB7431:
	.loc 5 53 0
	movq	4(%rbx), %rax
.LBE7431:
.LBE7430:
	.loc 1 583 0
	leaq	-4(%r12), %rdx
.LBB7433:
	.loc 1 585 0
	movq	-104(%rbp), %r8
.LBE7433:
.LBB7590:
.LBB7432:
	.loc 5 53 0
	movq	%rax, (%r12)
	movq	12(%rbx), %rax
	movq	%rax, 8(%r12)
	movq	20(%rbx), %rax
	movq	%rax, 16(%r12)
	movq	28(%rbx), %rax
	movq	%rax, 24(%r12)
.LVL874:
.LBE7432:
.LBE7590:
.LBB7591:
	.loc 1 585 0
	movl	76(%r15), %eax
.LVL875:
	movl	%eax, %esi
	subl	$1, %esi
.LVL876:
	js	.L351
	cmpl	$3, %eax
	jle	.L352
	movss	12(%r12), %xmm3
	movss	8(%r12), %xmm5
	movss	28(%r12), %xmm2
	movss	24(%r12), %xmm4
.LVL877:
	.p2align 4,,10
	.p2align 3
.L350:
.LBB7434:
	.loc 1 596 0 discriminator 2
	movss	8(%rdx), %xmm6
	movss	24(%rdx), %xmm7
.LBB7435:
.LBB7436:
	.loc 4 946 0 discriminator 2
	movaps	%xmm6, %xmm0
	movaps	%xmm7, %xmm1
	unpcklps	%xmm3, %xmm0
.LBE7436:
.LBE7435:
	.loc 1 596 0 discriminator 2
	movss	4(%rdx), %xmm3
.LBB7447:
.LBB7437:
	.loc 4 946 0 discriminator 2
	unpcklps	%xmm2, %xmm1
.LBE7437:
.LBE7447:
.LBB7448:
.LBB7449:
	movss	24(%rbx), %xmm2
	insertps	$0x10, 32(%rbx), %xmm2
.LBE7449:
.LBE7448:
.LBB7460:
.LBB7438:
	movlhps	%xmm1, %xmm0
.LBE7438:
.LBE7460:
.LBB7461:
.LBB7450:
	movss	8(%rbx), %xmm1
	insertps	$0x10, 16(%rbx), %xmm1
.LBE7450:
.LBE7461:
.LBB7462:
.LBB7463:
	.loc 4 189 0 discriminator 2
	mulps	48(%r14), %xmm0
.LBE7463:
.LBE7462:
.LBB7471:
.LBB7451:
	.loc 4 946 0 discriminator 2
	movlhps	%xmm2, %xmm1
.LBE7451:
.LBE7471:
	.loc 1 596 0 discriminator 2
	movss	20(%rdx), %xmm2
.LBB7472:
.LBB7464:
	.loc 4 189 0 discriminator 2
	mulps	16(%r14), %xmm1
	subps	%xmm1, %xmm0
.LBE7464:
.LBE7472:
.LBB7473:
.LBB7474:
.LBB7475:
.LBB7476:
.LBB7477:
.LBB7478:
	.loc 3 63 0 discriminator 2
	pxor	%xmm1, %xmm1
.LBE7478:
.LBE7477:
.LBE7476:
.LBE7475:
.LBE7474:
.LBE7473:
.LBB7532:
.LBB7533:
	.loc 7 58 0 discriminator 2
	haddps	%xmm0, %xmm0
.LBE7533:
.LBE7532:
.LBB7537:
.LBB7538:
	haddps	%xmm0, %xmm0
.LBE7538:
.LBE7537:
.LBB7542:
.LBB7543:
	.loc 4 960 0 discriminator 2
	movss	%xmm0, (%rbx)
.LBE7543:
.LBE7542:
.LBB7547:
.LBB7523:
.LBB7514:
.LBB7505:
.LBB7488:
.LBB7479:
	.loc 3 63 0 discriminator 2
	movzbl	0(%r13), %eax
	cvtsi2ss	%eax, %xmm1
	addss	%xmm1, %xmm0
.LBE7479:
.LBE7488:
.LBE7505:
.LBE7514:
.LBE7523:
.LBE7547:
.LBB7548:
.LBB7439:
	.loc 4 946 0 discriminator 2
	movaps	%xmm2, %xmm1
	unpcklps	%xmm4, %xmm1
.LBE7439:
.LBE7548:
.LBB7549:
.LBB7524:
.LBB7515:
.LBB7506:
.LBB7489:
.LBB7480:
	.loc 3 63 0 discriminator 2
	cvtss2sd	%xmm0, %xmm0
.LBE7480:
.LBE7489:
.LBB7490:
.LBB7491:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm0, %ecx
.LBE7491:
.LBE7490:
.LBE7506:
.LBE7515:
.LBE7524:
.LBE7549:
.LBB7550:
.LBB7440:
	.loc 4 946 0 discriminator 2
	movaps	%xmm3, %xmm0
	unpcklps	%xmm5, %xmm0
.LBE7440:
.LBE7550:
	.loc 1 596 0 discriminator 2
	movss	(%rdx), %xmm5
.LBB7551:
.LBB7441:
	.loc 4 946 0 discriminator 2
	movlhps	%xmm1, %xmm0
.LBE7441:
.LBE7551:
	.loc 1 604 0 discriminator 2
	testl	%ecx, %ecx
	setg	%al
	negl	%eax
	cmpl	$255, %ecx
	cmovbe	%ecx, %eax
	movb	%al, 0(%r13)
.LBB7552:
.LBB7452:
	.loc 4 946 0 discriminator 2
	movss	20(%rbx), %xmm4
	movss	4(%rbx), %xmm1
	insertps	$0x10, 28(%rbx), %xmm4
	insertps	$0x10, 12(%rbx), %xmm1
.LBE7452:
.LBE7552:
.LBB7553:
.LBB7465:
	.loc 4 189 0 discriminator 2
	mulps	48(%r14), %xmm0
.LBE7465:
.LBE7553:
.LBB7554:
.LBB7453:
	.loc 4 946 0 discriminator 2
	movlhps	%xmm4, %xmm1
.LBE7453:
.LBE7554:
	.loc 1 596 0 discriminator 2
	movss	16(%rdx), %xmm4
.LBB7555:
.LBB7466:
	.loc 4 189 0 discriminator 2
	mulps	16(%r14), %xmm1
	subps	%xmm1, %xmm0
.LBE7466:
.LBE7555:
.LBB7556:
.LBB7525:
.LBB7516:
.LBB7507:
.LBB7495:
.LBB7481:
	.loc 3 63 0 discriminator 2
	pxor	%xmm1, %xmm1
.LBE7481:
.LBE7495:
.LBE7507:
.LBE7516:
.LBE7525:
.LBE7556:
.LBB7557:
.LBB7534:
	.loc 7 58 0 discriminator 2
	haddps	%xmm0, %xmm0
.LBE7534:
.LBE7557:
.LBB7558:
.LBB7539:
	haddps	%xmm0, %xmm0
.LBE7539:
.LBE7558:
.LBB7559:
.LBB7544:
	.loc 4 960 0 discriminator 2
	movss	%xmm0, -4(%rbx)
.LBE7544:
.LBE7559:
.LBB7560:
.LBB7526:
.LBB7517:
.LBB7508:
.LBB7496:
.LBB7482:
	.loc 3 63 0 discriminator 2
	movzbl	-1(%r13), %eax
	cvtsi2ss	%eax, %xmm1
	addss	%xmm1, %xmm0
.LBE7482:
.LBE7496:
.LBE7508:
.LBE7517:
.LBE7526:
.LBE7560:
.LBB7561:
.LBB7442:
	.loc 4 946 0 discriminator 2
	movaps	%xmm4, %xmm1
	unpcklps	%xmm7, %xmm1
.LBE7442:
.LBE7561:
.LBB7562:
.LBB7527:
.LBB7518:
.LBB7509:
.LBB7497:
.LBB7483:
	.loc 3 63 0 discriminator 2
	cvtss2sd	%xmm0, %xmm0
.LBE7483:
.LBE7497:
.LBB7498:
.LBB7492:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm0, %ecx
.LBE7492:
.LBE7498:
.LBE7509:
.LBE7518:
.LBE7527:
.LBE7562:
.LBB7563:
.LBB7443:
	.loc 4 946 0 discriminator 2
	movaps	%xmm5, %xmm0
	unpcklps	%xmm6, %xmm0
	movlhps	%xmm1, %xmm0
.LBE7443:
.LBE7563:
	.loc 1 604 0 discriminator 2
	testl	%ecx, %ecx
	setg	%al
	negl	%eax
	cmpl	$255, %ecx
	cmovbe	%ecx, %eax
	movb	%al, -1(%r13)
.LBB7564:
.LBB7454:
	.loc 4 946 0 discriminator 2
	movss	16(%rbx), %xmm6
	movss	(%rbx), %xmm1
	insertps	$0x10, 24(%rbx), %xmm6
	insertps	$0x10, 8(%rbx), %xmm1
.LBE7454:
.LBE7564:
.LBB7565:
.LBB7467:
	.loc 4 189 0 discriminator 2
	mulps	48(%r14), %xmm0
.LBE7467:
.LBE7565:
.LBB7566:
.LBB7455:
	.loc 4 946 0 discriminator 2
	movlhps	%xmm6, %xmm1
.LBE7455:
.LBE7566:
.LBB7567:
.LBB7468:
	.loc 4 189 0 discriminator 2
	mulps	16(%r14), %xmm1
	subps	%xmm1, %xmm0
.LBE7468:
.LBE7567:
.LBB7568:
.LBB7528:
.LBB7519:
.LBB7510:
.LBB7499:
.LBB7484:
	.loc 3 63 0 discriminator 2
	pxor	%xmm1, %xmm1
.LBE7484:
.LBE7499:
.LBE7510:
.LBE7519:
.LBE7528:
.LBE7568:
.LBB7569:
.LBB7535:
	.loc 7 58 0 discriminator 2
	haddps	%xmm0, %xmm0
.LBE7535:
.LBE7569:
.LBB7570:
.LBB7540:
	haddps	%xmm0, %xmm0
.LBE7540:
.LBE7570:
.LBB7571:
.LBB7545:
	.loc 4 960 0 discriminator 2
	movss	%xmm0, -8(%rbx)
.LBE7545:
.LBE7571:
.LBB7572:
.LBB7529:
.LBB7520:
.LBB7511:
.LBB7500:
.LBB7485:
	.loc 3 63 0 discriminator 2
	movzbl	-2(%r13), %eax
	cvtsi2ss	%eax, %xmm1
	addss	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
.LBE7485:
.LBE7500:
.LBB7501:
.LBB7493:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm0, %ecx
.LBE7493:
.LBE7501:
.LBE7511:
.LBE7520:
.LBE7529:
.LBE7572:
	.loc 1 604 0 discriminator 2
	testl	%ecx, %ecx
	setg	%al
	negl	%eax
	cmpl	$255, %ecx
	cmovbe	%ecx, %eax
.LBE7434:
	.loc 1 585 0 discriminator 2
	subq	$12, %rdx
	subq	$12, %rbx
.LBB7587:
	.loc 1 604 0 discriminator 2
	movb	%al, -2(%r13)
	movl	%esi, %eax
.LBE7587:
	.loc 1 585 0 discriminator 2
	subq	$3, %r13
	subl	$4, %eax
	subl	$3, %esi
.LVL878:
	cmpl	$1, %eax
	jg	.L350
.LVL879:
	.p2align 4,,10
	.p2align 3
.L352:
.LBB7588:
.LBB7573:
.LBB7444:
	.loc 4 946 0
	movss	24(%rdx), %xmm1
	movss	8(%rdx), %xmm0
	insertps	$0x10, 32(%rdx), %xmm1
.LBE7444:
.LBE7573:
.LBB7574:
.LBB7456:
	movss	24(%rbx), %xmm2
.LBE7456:
.LBE7574:
.LBB7575:
.LBB7445:
	insertps	$0x10, 16(%rdx), %xmm0
.LBE7445:
.LBE7575:
.LBB7576:
.LBB7457:
	insertps	$0x10, 32(%rbx), %xmm2
.LBE7457:
.LBE7576:
.LBB7577:
.LBB7446:
	movlhps	%xmm1, %xmm0
.LBE7446:
.LBE7577:
.LBB7578:
.LBB7458:
	movss	8(%rbx), %xmm1
	insertps	$0x10, 16(%rbx), %xmm1
.LBE7458:
.LBE7578:
.LBB7579:
.LBB7469:
	.loc 4 189 0
	mulps	48(%r14), %xmm0
.LBE7469:
.LBE7579:
.LBB7580:
.LBB7459:
	.loc 4 946 0
	movlhps	%xmm2, %xmm1
.LBE7459:
.LBE7580:
.LBB7581:
.LBB7470:
	.loc 4 189 0
	mulps	16(%r14), %xmm1
	subps	%xmm1, %xmm0
.LBE7470:
.LBE7581:
.LBB7582:
.LBB7530:
.LBB7521:
.LBB7512:
.LBB7502:
.LBB7486:
	.loc 3 63 0
	pxor	%xmm1, %xmm1
.LBE7486:
.LBE7502:
.LBE7512:
.LBE7521:
.LBE7530:
.LBE7582:
.LBB7583:
.LBB7536:
	.loc 7 58 0
	haddps	%xmm0, %xmm0
.LBE7536:
.LBE7583:
.LBB7584:
.LBB7541:
	haddps	%xmm0, %xmm0
.LBE7541:
.LBE7584:
.LBB7585:
.LBB7546:
	.loc 4 960 0
	movss	%xmm0, (%rbx)
.LBE7546:
.LBE7585:
.LBB7586:
.LBB7531:
.LBB7522:
.LBB7513:
.LBB7503:
.LBB7487:
	.loc 3 63 0
	movzbl	0(%r13), %eax
	cvtsi2ss	%eax, %xmm1
	addss	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
.LBE7487:
.LBE7503:
.LBB7504:
.LBB7494:
	.loc 3 827 0
	cvtsd2si	%xmm0, %ecx
.LBE7494:
.LBE7504:
.LBE7513:
.LBE7522:
.LBE7531:
.LBE7586:
	.loc 1 604 0
	testl	%ecx, %ecx
	setg	%al
	negl	%eax
	cmpl	$255, %ecx
	cmovbe	%ecx, %eax
.LBE7588:
	.loc 1 585 0
	subl	$1, %esi
.LVL880:
	subq	$4, %rdx
.LVL881:
.LBB7589:
	.loc 1 604 0
	movb	%al, 0(%r13)
.LBE7589:
	.loc 1 585 0
	subq	$4, %rbx
.LVL882:
	subq	$1, %r13
.LVL883:
	cmpl	$-1, %esi
	jne	.L352
.L351:
	addl	$1, -88(%rbp)
.LVL884:
	addq	$1, %r8
.LBE7591:
	movq	-96(%rbp), %rsp
	movl	-88(%rbp), %eax
.LVL885:
	cmpl	%eax, -152(%rbp)
	jg	.L347
	movl	-168(%rbp), %edi
	movl	-172(%rbp), %ebx
.LVL886:
	movl	%edi, %eax
.LVL887:
	addl	-192(%rbp), %eax
	movl	%edi, -88(%rbp)
.LVL888:
	cmpl	%ebx, %eax
	cmovg	%ebx, %eax
	movl	%eax, -152(%rbp)
	movl	%edi, %eax
	movl	-184(%rbp), %edi
	addl	%edi, %eax
	movl	%eax, -168(%rbp)
	subl	%edi, %eax
	cmpl	%eax, %ebx
	jg	.L348
.LVL889:
.L344:
	call	GOMP_barrier
.LVL890:
.LBE7334:
.LBE7333:
.LBE7332:
	.loc 1 392 0
	movq	-56(%rbp), %rax
	xorq	%fs:40, %rax
	jne	.L407
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.LVL891:
	.p2align 4,,10
	.p2align 3
.L397:
	.cfi_restore_state
.LBB7595:
.LBB7331:
.LBB7330:
.LBB7141:
.LBB7140:
	.loc 1 405 0
	movl	-172(%rbp), %ecx
.LBE7140:
	.loc 1 404 0
	xorl	%r9d, %r9d
	jmp	.L319
.LVL892:
.L406:
.LBE7141:
.LBE7330:
.LBE7331:
.LBE7595:
.LBB7596:
.LBB7594:
.LBB7593:
.LBB7592:
	.loc 1 556 0
	movq	%rcx, %rax
	jmp	.L354
.LVL893:
.L407:
.LBE7592:
.LBE7593:
.LBE7594:
.LBE7596:
	.loc 1 392 0
	call	__stack_chk_fail
.LVL894:
	.cfi_endproc
.LFE12403:
	.size	_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_._omp_fn.4, .-_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_._omp_fn.4
	.section	.text.unlikely._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_._omp_fn.4,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_,comdat
.LCOLDE4:
	.section	.text._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_._omp_fn.4,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_,comdat
.LHOTE4:
	.section	.text.unlikely._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_._omp_fn.5,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_,comdat
.LCOLDB5:
	.section	.text._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_._omp_fn.5,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_,comdat
.LHOTB5:
	.p2align 4,,15
	.type	_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_._omp_fn.5, @function
_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_._omp_fn.5:
.LFB12404:
	.loc 1 392 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
.LVL895:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movl	$1, %ecx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	leaq	-64(%rbp), %r9
	pushq	%rbx
	leaq	-72(%rbp), %r8
	subq	$216, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movslq	76(%rdi), %rsi
	.loc 1 392 0
	movq	16(%rdi), %r14
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
	movq	24(%rdi), %rax
	movq	%rdi, -88(%rbp)
	movq	%rax, -232(%rbp)
.LVL896:
	movslq	80(%rdi), %rax
.LVL897:
	xorl	%edi, %edi
.LVL898:
	movq	%rax, %rdx
	movl	%eax, -96(%rbp)
.LVL899:
	movq	%rax, %rbx
	movq	%rax, -248(%rbp)
	call	GOMP_loop_dynamic_start
.LVL900:
	testb	%al, %al
	movq	-88(%rbp), %r10
	je	.L409
	movq	%rbx, %rax
	movq	%r14, %r13
	salq	$2, %rax
	movq	%rax, -240(%rbp)
	leaq	96(%r14), %rax
	movq	%r10, %r14
.LVL901:
	movq	%rax, -224(%rbp)
.LVL902:
.L413:
	movq	-72(%rbp), %rax
	movl	-64(%rbp), %edi
	movq	%r14, %r15
	movq	%r13, %r14
	movl	%eax, -172(%rbp)
.LVL903:
	movl	%eax, -176(%rbp)
	cltq
.LVL904:
	movq	%rax, -192(%rbp)
	salq	$2, %rax
	movl	%edi, -216(%rbp)
	movq	%rax, -184(%rbp)
.L412:
.LVL905:
.LBB7597:
.LBB7598:
.LBB7599:
	.loc 1 399 0
	movl	72(%r15), %r9d
	movq	-232(%rbp), %rax
	xorl	%esi, %esi
	addq	-184(%rbp), %rax
	movq	$0, -104(%rbp)
	testl	%r9d, %r9d
	movq	%rax, -120(%rbp)
	jle	.L436
.LVL906:
	.p2align 4,,10
	.p2align 3
.L493:
.LBB7600:
	.loc 1 400 0
	movq	32(%r15), %rax
	movq	-104(%rbp), %rbx
	movq	16(%rax), %rdi
.LVL907:
.LBB7601:
.LBB7602:
	.loc 2 436 0
	movq	72(%rax), %rax
.LVL908:
.LBE7602:
.LBE7601:
	.loc 1 400 0
	movq	%rbx, %r8
	imulq	(%rax), %r8
	movq	-192(%rbp), %rax
	addq	%rax, %r8
	addq	%rdi, %r8
.LVL909:
	.loc 1 401 0
	addq	%rax, %rdi
.LVL910:
	.loc 1 402 0
	movq	48(%r15), %rax
.LVL911:
.LBB7603:
.LBB7604:
	.loc 2 430 0
	movq	72(%rax), %rdx
.LBE7604:
.LBE7603:
	.loc 1 402 0
	imulq	(%rdx), %rbx
	movq	%rbx, %rdx
	addq	-184(%rbp), %rdx
	addq	16(%rax), %rdx
.LVL912:
.LBB7605:
	.loc 1 405 0
	movl	-96(%rbp), %eax
.LVL913:
	testl	%eax, %eax
	jle	.L433
	movl	76(%r15), %eax
	leal	-15(%rax), %ecx
	cmpl	%ecx, -172(%rbp)
	jge	.L512
	leaq	80(%r14), %r13
	leaq	64(%r14), %r12
	movq	%r14, -112(%rbp)
	movq	-120(%rbp), %rbx
	movl	-176(%rbp), %r14d
	xorl	%r9d, %r9d
.LVL914:
	.p2align 4,,10
	.p2align 3
.L451:
	pxor	%xmm1, %xmm1
	movl	$1, %ecx
	movdqa	(%r15), %xmm5
	xorl	%eax, %eax
	movl	%r9d, -88(%rbp)
	movaps	%xmm1, %xmm7
	movaps	%xmm1, %xmm6
	movaps	%xmm1, %xmm8
.LVL915:
.L456:
.LBB7606:
.LBB7607:
.LBB7608:
	.loc 1 409 0
	cmpl	%eax, %esi
	movl	%eax, %r11d
.LVL916:
	jge	.L453
	movq	%rdi, %r10
.L458:
.LVL917:
.LBB7609:
.LBB7610:
	.loc 3 698 0 discriminator 4
	movdqu	(%r10), %xmm2
.LBE7610:
.LBE7609:
.LBB7611:
.LBB7612:
.LBB7613:
.LBB7614:
	.loc 4 884 0 discriminator 4
	movss	0(%r13,%rax,4), %xmm0
.LVL918:
.LBE7614:
.LBE7613:
.LBE7612:
.LBE7611:
	.loc 1 424 0 discriminator 4
	cmpl	%r11d, %esi
.LBB7615:
.LBB7616:
	.loc 3 965 0 discriminator 4
	movdqa	%xmm2, %xmm3
.LBE7616:
.LBE7615:
.LBB7618:
.LBB7619:
	.loc 4 743 0 discriminator 4
	shufps	$0, %xmm0, %xmm0
.LVL919:
.LBE7619:
.LBE7618:
.LBB7620:
.LBB7617:
	.loc 3 965 0 discriminator 4
	punpckhbw	%xmm5, %xmm3
.LVL920:
.LBE7617:
.LBE7620:
.LBB7621:
.LBB7622:
	.loc 3 989 0 discriminator 4
	punpcklbw	%xmm5, %xmm2
.LVL921:
.LBE7622:
.LBE7621:
.LBB7623:
.LBB7624:
	.loc 3 995 0 discriminator 4
	movdqa	%xmm3, %xmm9
.LBE7624:
.LBE7623:
.LBB7626:
.LBB7627:
	.loc 3 971 0 discriminator 4
	punpckhwd	%xmm5, %xmm3
.LVL922:
.LBE7627:
.LBE7626:
.LBB7628:
.LBB7629:
	movdqa	%xmm2, %xmm4
.LBE7629:
.LBE7628:
.LBB7631:
.LBB7625:
	.loc 3 995 0 discriminator 4
	punpcklwd	%xmm5, %xmm9
.LBE7625:
.LBE7631:
.LBB7632:
.LBB7630:
	.loc 3 971 0 discriminator 4
	punpckhwd	%xmm5, %xmm4
.LVL923:
.LBE7630:
.LBE7632:
.LBB7633:
.LBB7634:
	.loc 3 767 0 discriminator 4
	cvtdq2ps	%xmm3, %xmm3
.LBE7634:
.LBE7633:
.LBB7635:
.LBB7636:
	.loc 4 183 0 discriminator 4
	mulps	%xmm0, %xmm3
.LBE7636:
.LBE7635:
.LBB7638:
.LBB7639:
	.loc 3 995 0 discriminator 4
	punpcklwd	%xmm5, %xmm2
.LVL924:
.LBE7639:
.LBE7638:
.LBB7640:
.LBB7641:
	.loc 3 767 0 discriminator 4
	cvtdq2ps	%xmm4, %xmm4
.LVL925:
.LBE7641:
.LBE7640:
.LBB7642:
.LBB7643:
	.loc 4 183 0 discriminator 4
	mulps	%xmm0, %xmm4
.LBE7643:
.LBE7642:
.LBB7645:
.LBB7637:
	addps	%xmm3, %xmm8
.LVL926:
.LBE7637:
.LBE7645:
.LBB7646:
.LBB7647:
	.loc 3 767 0 discriminator 4
	cvtdq2ps	%xmm2, %xmm2
.LVL927:
.LBE7647:
.LBE7646:
.LBB7648:
.LBB7649:
	cvtdq2ps	%xmm9, %xmm3
.LVL928:
.LBE7649:
.LBE7648:
.LBB7650:
.LBB7651:
	.loc 4 183 0 discriminator 4
	mulps	%xmm0, %xmm3
.LBE7651:
.LBE7650:
.LBB7653:
.LBB7654:
	mulps	%xmm2, %xmm0
.LVL929:
.LBE7654:
.LBE7653:
.LBB7656:
.LBB7644:
	addps	%xmm4, %xmm7
.LVL930:
.LBE7644:
.LBE7656:
.LBB7657:
.LBB7652:
	addps	%xmm3, %xmm6
.LVL931:
.LBE7652:
.LBE7657:
.LBB7658:
.LBB7655:
	addps	%xmm0, %xmm1
.LVL932:
.LBE7655:
.LBE7658:
	.loc 1 424 0 discriminator 4
	jg	.L454
	movq	%rbx, %r10
.LVL933:
.L457:
.LBB7659:
.LBB7660:
.LBB7661:
.LBB7662:
	.loc 4 884 0 discriminator 4
	movss	(%r12,%rax,4), %xmm0
.LVL934:
.LBE7662:
.LBE7661:
.LBE7660:
.LBE7659:
.LBB7663:
.LBB7664:
	.loc 4 931 0 discriminator 4
	movups	48(%r10), %xmm9
.LVL935:
	addq	$1, %rax
.LVL936:
	addl	$1, %ecx
.LBE7664:
.LBE7663:
.LBB7665:
.LBB7666:
	.loc 4 743 0 discriminator 4
	shufps	$0, %xmm0, %xmm0
.LVL937:
.LBE7666:
.LBE7665:
.LBB7667:
.LBB7668:
	.loc 4 931 0 discriminator 4
	movups	32(%r10), %xmm4
.LVL938:
.LBE7668:
.LBE7667:
.LBE7608:
	.loc 1 408 0 discriminator 4
	cmpq	$4, %rax
.LBB7689:
.LBB7669:
.LBB7670:
	.loc 4 931 0 discriminator 4
	movups	16(%r10), %xmm3
.LVL939:
.LBE7670:
.LBE7669:
.LBB7671:
.LBB7672:
	.loc 4 189 0 discriminator 4
	mulps	%xmm0, %xmm9
.LVL940:
.LBE7672:
.LBE7671:
.LBB7674:
.LBB7675:
	.loc 4 931 0 discriminator 4
	movups	(%r10), %xmm2
.LVL941:
.LBE7675:
.LBE7674:
.LBB7676:
.LBB7677:
	.loc 4 189 0 discriminator 4
	mulps	%xmm0, %xmm4
.LVL942:
.LBE7677:
.LBE7676:
.LBB7679:
.LBB7680:
	mulps	%xmm0, %xmm3
.LVL943:
.LBE7680:
.LBE7679:
.LBB7682:
.LBB7673:
	subps	%xmm9, %xmm8
.LVL944:
.LBE7673:
.LBE7682:
.LBB7683:
.LBB7684:
	mulps	%xmm2, %xmm0
.LVL945:
.LBE7684:
.LBE7683:
.LBB7686:
.LBB7678:
	subps	%xmm4, %xmm6
.LVL946:
.LBE7678:
.LBE7686:
.LBB7687:
.LBB7681:
	subps	%xmm3, %xmm7
.LVL947:
.LBE7681:
.LBE7687:
.LBB7688:
.LBB7685:
	subps	%xmm0, %xmm1
.LVL948:
.LBE7685:
.LBE7688:
.LBE7689:
	.loc 1 408 0 discriminator 4
	jne	.L456
	movl	-88(%rbp), %r9d
.LVL949:
.LBE7607:
.LBE7606:
	.loc 1 405 0
	addq	$16, %r8
.LVL950:
	addq	$16, %rdi
.LVL951:
.LBB7700:
.LBB7691:
.LBB7692:
	.loc 4 980 0
	movups	%xmm1, (%rdx)
.LVL952:
.LBE7692:
.LBE7691:
.LBE7700:
	.loc 1 405 0
	addq	$64, %rbx
.LVL953:
	addq	$64, %rdx
.LVL954:
.LBB7701:
.LBB7693:
.LBB7694:
	.loc 4 980 0
	movups	%xmm7, -48(%rdx)
.LVL955:
.LBE7694:
.LBE7693:
.LBE7701:
	.loc 1 405 0
	addl	$16, %r9d
.LBB7702:
.LBB7695:
.LBB7696:
	.loc 4 980 0
	movups	%xmm6, -32(%rdx)
.LVL956:
.LBE7696:
.LBE7695:
.LBB7697:
.LBB7698:
	movups	%xmm8, -16(%rdx)
.LVL957:
.LBE7698:
.LBE7697:
.LBE7702:
	.loc 1 405 0
	cmpl	%r9d, -96(%rbp)
	jle	.L513
	.loc 1 405 0 is_stmt 0 discriminator 1
	movl	76(%r15), %eax
	leal	(%r9,%r14), %ecx
	leal	-15(%rax), %r10d
.LVL958:
	cmpl	%ecx, %r10d
	jg	.L451
	movq	-112(%rbp), %r14
.LVL959:
.L434:
.LBE7605:
.LBB7704:
	.loc 1 442 0 is_stmt 1
	cmpl	%ecx, %eax
	jle	.L507
.LBB7705:
	.loc 1 450 0 discriminator 1
	movslq	%eax, %rbx
	.loc 1 445 0 discriminator 1
	leal	(%rax,%rax), %ecx
	movq	%r15, -112(%rbp)
	.loc 1 450 0 discriminator 1
	leaq	0(,%rbx,4), %r10
	.loc 1 445 0 discriminator 1
	movq	%rbx, -88(%rbp)
	movl	-176(%rbp), %r15d
	movslq	%ecx, %r12
	addl	%eax, %ecx
	movl	-96(%rbp), %ebx
	.loc 1 450 0 discriminator 1
	negq	%r10
	.loc 1 445 0 discriminator 1
	movslq	%ecx, %r13
	jmp	.L482
.LVL960:
	.p2align 4,,10
	.p2align 3
.L514:
	.loc 1 445 0 is_stmt 0
	movzbl	(%r8), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2ss	%ecx, %xmm0
	mulss	80(%r14), %xmm0
	addss	(%rdx), %xmm0
	movss	%xmm0, (%rdx)
	.loc 1 449 0 is_stmt 1
	je	.L441
	.loc 1 450 0
	leaq	(%rdx,%r10), %rcx
	movss	64(%r14), %xmm1
	.loc 1 445 0
	movq	%r8, %r11
	subq	-88(%rbp), %r11
	.loc 1 449 0
	cmpl	$1, %esi
	.loc 1 450 0
	mulss	(%rcx), %xmm1
	subss	%xmm1, %xmm0
	.loc 1 445 0
	pxor	%xmm1, %xmm1
	.loc 1 450 0
	movss	%xmm0, (%rdx)
.LVL961:
	.loc 1 445 0
	movzbl	(%r11), %r11d
	cvtsi2ss	%r11d, %xmm1
	mulss	84(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
	.loc 1 449 0
	je	.L443
	.loc 1 450 0
	movss	68(%r14), %xmm1
	addq	%r10, %rcx
	.loc 1 445 0
	movq	%r8, %r11
	.loc 1 450 0
	mulss	(%rcx), %xmm1
	.loc 1 445 0
	subq	%r12, %r11
	.loc 1 449 0
	cmpl	$2, %esi
	.loc 1 450 0
	subss	%xmm1, %xmm0
	.loc 1 445 0
	pxor	%xmm1, %xmm1
	.loc 1 450 0
	movss	%xmm0, (%rdx)
.LVL962:
	.loc 1 445 0
	movzbl	(%r11), %r11d
	cvtsi2ss	%r11d, %xmm1
	mulss	88(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
	.loc 1 449 0
	je	.L445
	.loc 1 450 0
	movss	72(%r14), %xmm1
	.loc 1 445 0
	movq	%r8, %r11
	.loc 1 450 0
	mulss	(%rcx,%r10), %xmm1
	.loc 1 445 0
	subq	%r13, %r11
	.loc 1 449 0
	cmpl	$3, %esi
	.loc 1 450 0
	subss	%xmm1, %xmm0
	.loc 1 445 0
	pxor	%xmm1, %xmm1
	.loc 1 450 0
	movss	%xmm0, (%rdx)
.LVL963:
	.loc 1 445 0
	movzbl	(%r11), %r11d
	cvtsi2ss	%r11d, %xmm1
	mulss	92(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
	.loc 1 449 0
	je	.L447
	.loc 1 450 0
	movss	(%rcx,%r10,2), %xmm1
	mulss	76(%r14), %xmm1
	subss	%xmm1, %xmm0
.L448:
.LBE7705:
	.loc 1 442 0
	addl	$1, %r9d
.LVL964:
	movss	%xmm0, (%rdx)
.LVL965:
	addq	$1, %r8
.LVL966:
	addq	$1, %rdi
.LVL967:
	addq	$4, %rdx
.LVL968:
	cmpl	%r9d, %ebx
	jle	.L508
	.loc 1 442 0 is_stmt 0 discriminator 1
	leal	(%r9,%r15), %ecx
	cmpl	%ecx, %eax
	jle	.L508
.LVL969:
.L482:
.LBB7706:
	.loc 1 444 0 is_stmt 1 discriminator 1
	testl	%esi, %esi
	jns	.L514
	.loc 1 447 0
	movzbl	(%rdi), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2ss	%ecx, %xmm0
	mulss	80(%r14), %xmm0
	addss	(%rdx), %xmm0
	movss	%xmm0, (%rdx)
.L441:
	.loc 1 452 0
	movzbl	(%rdi), %ecx
	pxor	%xmm1, %xmm1
	cvtsi2ss	%ecx, %xmm1
	mulss	64(%r14), %xmm1
	subss	%xmm1, %xmm0
	.loc 1 447 0
	pxor	%xmm1, %xmm1
	.loc 1 452 0
	movss	%xmm0, (%rdx)
.LVL970:
	.loc 1 447 0
	movzbl	(%rdi), %ecx
	cvtsi2ss	%ecx, %xmm1
	mulss	84(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
.L443:
	.loc 1 452 0
	movzbl	(%rdi), %ecx
	pxor	%xmm1, %xmm1
	cvtsi2ss	%ecx, %xmm1
	mulss	68(%r14), %xmm1
	subss	%xmm1, %xmm0
	.loc 1 447 0
	pxor	%xmm1, %xmm1
	.loc 1 452 0
	movss	%xmm0, (%rdx)
.LVL971:
	.loc 1 447 0
	movzbl	(%rdi), %ecx
	cvtsi2ss	%ecx, %xmm1
	mulss	88(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
.L445:
	.loc 1 452 0
	movzbl	(%rdi), %ecx
	pxor	%xmm1, %xmm1
	cvtsi2ss	%ecx, %xmm1
	mulss	72(%r14), %xmm1
	subss	%xmm1, %xmm0
	.loc 1 447 0
	pxor	%xmm1, %xmm1
	.loc 1 452 0
	movss	%xmm0, (%rdx)
.LVL972:
	.loc 1 447 0
	movzbl	(%rdi), %ecx
	cvtsi2ss	%ecx, %xmm1
	mulss	92(%r14), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
.L447:
	.loc 1 452 0
	movzbl	(%rdi), %ecx
	pxor	%xmm1, %xmm1
	cvtsi2ss	%ecx, %xmm1
	mulss	76(%r14), %xmm1
	subss	%xmm1, %xmm0
	jmp	.L448
.LVL973:
	.p2align 4,,10
	.p2align 3
.L454:
.LBE7706:
.LBE7704:
.LBB7707:
.LBB7703:
.LBB7699:
.LBB7690:
	.loc 1 424 0 discriminator 1
	movl	76(%r15), %r10d
.LVL974:
	movq	%rdx, %r11
	imull	%ecx, %r10d
	movslq	%r10d, %r10
	salq	$2, %r10
	subq	%r10, %r11
	movq	%r11, %r10
	jmp	.L457
.LVL975:
	.p2align 4,,10
	.p2align 3
.L453:
	.loc 1 409 0 discriminator 1
	movl	76(%r15), %r10d
	movq	%r8, %r9
	imull	%eax, %r10d
	movslq	%r10d, %r10
	subq	%r10, %r9
	movq	%r9, %r10
	jmp	.L458
.LVL976:
	.p2align 4,,10
	.p2align 3
.L513:
	movq	-112(%rbp), %r14
.LVL977:
.L507:
	movl	72(%r15), %r9d
.LVL978:
.L433:
.LBE7690:
.LBE7699:
.LBE7703:
.LBE7707:
.LBE7600:
	.loc 1 399 0 discriminator 2
	addl	$1, %esi
.LVL979:
	addq	$1, -104(%rbp)
	cmpl	%r9d, %esi
	jl	.L493
.LVL980:
.L436:
.LBE7599:
.LBB7710:
	.loc 1 457 0
	leal	-1(%r9), %edx
	movslq	%edx, %rax
	testl	%edx, %edx
	movl	%edx, %ebx
.LVL981:
	movq	%rax, -200(%rbp)
	js	.L417
	movl	-96(%rbp), %r12d
	movl	-176(%rbp), %r10d
	movq	%rax, %r8
.LVL982:
	.p2align 4,,10
	.p2align 3
.L494:
.LBB7711:
	.loc 1 458 0
	movq	32(%r15), %rax
	movq	%r8, %rsi
	.loc 1 459 0
	movslq	%edx, %rdx
	.loc 1 458 0
	movq	-192(%rbp), %rcx
	.loc 1 460 0
	movq	-184(%rbp), %r11
	movq	16(%rax), %rdi
.LVL983:
.LBB7712:
.LBB7713:
	.loc 2 436 0
	movq	72(%rax), %rax
.LVL984:
	movq	(%rax), %rax
.LVL985:
.LBE7713:
.LBE7712:
	.loc 1 459 0
	imulq	%rax, %rdx
	.loc 1 458 0
	imulq	%rax, %rsi
	addq	%rcx, %rsi
	.loc 1 459 0
	addq	%rdx, %rcx
	.loc 1 460 0
	movq	64(%r15), %rdx
	.loc 1 458 0
	addq	%rdi, %rsi
.LVL986:
	.loc 1 459 0
	addq	%rdi, %rcx
.LVL987:
	.loc 1 460 0
	movq	%r8, %rdi
.LBB7714:
.LBB7715:
	.loc 2 430 0
	movq	72(%rdx), %rax
.LBE7715:
.LBE7714:
	.loc 1 460 0
	imulq	(%rax), %rdi
	movq	%rdi, %rax
	.loc 1 461 0
	movq	48(%r15), %rdi
	.loc 1 460 0
	addq	%r11, %rax
	addq	16(%rdx), %rax
.LVL988:
.LBB7716:
.LBB7717:
	.loc 2 430 0
	movq	72(%rdi), %rdx
.LBE7717:
.LBE7716:
	.loc 1 461 0
	imulq	(%rdx), %r8
	movq	%r8, %rdx
	addq	%r11, %rdx
	addq	16(%rdi), %rdx
.LVL989:
.LBB7718:
	.loc 1 463 0
	testl	%r12d, %r12d
	jle	.L414
	movl	76(%r15), %r8d
	leal	-15(%r8), %edi
.LVL990:
	cmpl	%edi, -172(%rbp)
	jge	.L473
	movq	-224(%rbp), %r13
	xorl	%edi, %edi
.LVL991:
	.p2align 4,,10
	.p2align 3
.L416:
	xorl	%r8d, %r8d
	pxor	%xmm0, %xmm0
.LBB7719:
.LBB7720:
.LBB7721:
	.loc 1 467 0
	addl	$1, %r8d
	movq	%r13, %r11
	subl	%r8d, %r9d
	movdqa	(%r15), %xmm4
.LVL992:
	cmpl	%ebx, %r9d
.LBE7721:
.LBE7720:
.LBE7719:
	.loc 1 463 0
	movaps	%xmm0, %xmm7
.LVL993:
	movaps	%xmm0, %xmm6
.LVL994:
	movaps	%xmm0, %xmm3
.LVL995:
.LBB7887:
.LBB7854:
.LBB7850:
	.loc 1 467 0
	jg	.L431
.LVL996:
.L515:
.LBB7722:
	.loc 1 496 0
	movss	(%r11), %xmm1
	addq	$4, %r11
.LBB7723:
.LBB7724:
	.loc 3 698 0
	movdqu	(%rcx), %xmm2
.LBE7724:
.LBE7723:
	.loc 1 496 0
	subss	-36(%r11), %xmm1
.LVL997:
.LBE7722:
.LBE7850:
	.loc 1 466 0
	cmpl	$4, %r8d
.LBB7851:
.LBB7765:
.LBB7725:
.LBB7726:
	.loc 3 965 0
	movdqa	%xmm2, %xmm5
.LBE7726:
.LBE7725:
.LBB7728:
.LBB7729:
	.loc 3 989 0
	punpcklbw	%xmm4, %xmm2
.LBE7729:
.LBE7728:
.LBB7730:
.LBB7727:
	.loc 3 965 0
	punpckhbw	%xmm4, %xmm5
.LBE7727:
.LBE7730:
.LBB7731:
.LBB7732:
	.loc 3 971 0
	movdqa	%xmm2, %xmm8
.LBE7732:
.LBE7731:
.LBB7734:
.LBB7735:
	.loc 3 995 0
	punpcklwd	%xmm4, %xmm2
.LBE7735:
.LBE7734:
.LBB7736:
.LBB7737:
	.loc 4 891 0
	shufps	$0, %xmm1, %xmm1
.LVL998:
.LBE7737:
.LBE7736:
.LBB7738:
.LBB7739:
	.loc 3 995 0
	movdqa	%xmm5, %xmm9
.LBE7739:
.LBE7738:
.LBB7741:
.LBB7742:
	.loc 3 971 0
	punpckhwd	%xmm4, %xmm5
.LVL999:
.LBE7742:
.LBE7741:
.LBB7743:
.LBB7744:
	.loc 3 767 0
	cvtdq2ps	%xmm2, %xmm2
.LBE7744:
.LBE7743:
.LBB7745:
.LBB7733:
	.loc 3 971 0
	punpckhwd	%xmm4, %xmm8
.LVL1000:
.LBE7733:
.LBE7745:
.LBB7746:
.LBB7740:
	.loc 3 995 0
	punpcklwd	%xmm4, %xmm9
.LVL1001:
.LBE7740:
.LBE7746:
.LBB7747:
.LBB7748:
	.loc 3 767 0
	cvtdq2ps	%xmm5, %xmm5
.LVL1002:
.LBE7748:
.LBE7747:
.LBB7749:
.LBB7750:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
	addps	%xmm5, %xmm3
.LVL1003:
.LBE7750:
.LBE7749:
.LBB7751:
.LBB7752:
	.loc 3 767 0
	cvtdq2ps	%xmm9, %xmm5
.LVL1004:
.LBE7752:
.LBE7751:
.LBB7753:
.LBB7754:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
	addps	%xmm5, %xmm6
.LVL1005:
.LBE7754:
.LBE7753:
.LBB7755:
.LBB7756:
	.loc 3 767 0
	cvtdq2ps	%xmm8, %xmm5
.LVL1006:
.LBE7756:
.LBE7755:
.LBB7757:
.LBB7758:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
.LBE7758:
.LBE7757:
.LBB7760:
.LBB7761:
	mulps	%xmm2, %xmm1
.LBE7761:
.LBE7760:
.LBB7763:
.LBB7759:
	addps	%xmm5, %xmm7
.LVL1007:
.LBE7759:
.LBE7763:
.LBB7764:
.LBB7762:
	addps	%xmm1, %xmm0
.LVL1008:
.LBE7762:
.LBE7764:
.LBE7765:
.LBE7851:
	.loc 1 466 0
	je	.L428
.LVL1009:
.L516:
	movl	72(%r15), %r9d
.LBB7852:
	.loc 1 467 0
	addl	$1, %r8d
.LVL1010:
	subl	%r8d, %r9d
	cmpl	%ebx, %r9d
	jle	.L515
.LVL1011:
.L431:
.LBB7766:
.LBB7767:
.LBB7768:
	.loc 3 698 0
	movl	76(%r15), %r9d
.LBE7768:
.LBE7767:
.LBB7770:
.LBB7771:
.LBB7772:
.LBB7773:
	.loc 4 884 0
	movss	(%r11), %xmm1
.LVL1012:
	addq	$4, %r11
.LBE7773:
.LBE7772:
.LBE7771:
.LBE7770:
.LBB7774:
.LBB7775:
	.loc 4 743 0
	shufps	$0, %xmm1, %xmm1
.LVL1013:
.LBE7775:
.LBE7774:
.LBB7776:
.LBB7769:
	.loc 3 698 0
	imull	%r8d, %r9d
	movslq	%r9d, %r9
	movdqu	(%rsi,%r9), %xmm2
.LVL1014:
.LBE7769:
.LBE7776:
.LBB7777:
.LBB7778:
	.loc 3 965 0
	movdqa	%xmm2, %xmm5
.LBE7778:
.LBE7777:
.LBB7780:
.LBB7781:
	.loc 3 989 0
	punpcklbw	%xmm4, %xmm2
.LVL1015:
.LBE7781:
.LBE7780:
.LBB7782:
.LBB7779:
	.loc 3 965 0
	punpckhbw	%xmm4, %xmm5
.LVL1016:
.LBE7779:
.LBE7782:
.LBB7783:
.LBB7784:
	.loc 3 971 0
	movdqa	%xmm2, %xmm8
.LBE7784:
.LBE7783:
.LBB7786:
.LBB7787:
	.loc 3 995 0
	movdqa	%xmm5, %xmm9
.LBE7787:
.LBE7786:
.LBB7789:
.LBB7790:
	.loc 3 971 0
	punpckhwd	%xmm4, %xmm5
.LVL1017:
.LBE7790:
.LBE7789:
.LBB7791:
.LBB7785:
	punpckhwd	%xmm4, %xmm8
.LVL1018:
.LBE7785:
.LBE7791:
.LBB7792:
.LBB7788:
	.loc 3 995 0
	punpcklwd	%xmm4, %xmm9
.LBE7788:
.LBE7792:
.LBB7793:
.LBB7794:
	.loc 3 767 0
	cvtdq2ps	%xmm5, %xmm5
.LBE7794:
.LBE7793:
.LBB7795:
.LBB7796:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
.LBE7796:
.LBE7795:
.LBB7798:
.LBB7799:
	.loc 3 995 0
	punpcklwd	%xmm4, %xmm2
.LVL1019:
.LBE7799:
.LBE7798:
.LBB7800:
.LBB7797:
	.loc 4 183 0
	addps	%xmm5, %xmm3
.LVL1020:
.LBE7797:
.LBE7800:
.LBB7801:
.LBB7802:
	.loc 3 767 0
	cvtdq2ps	%xmm9, %xmm5
.LVL1021:
.LBE7802:
.LBE7801:
.LBB7803:
.LBB7804:
	cvtdq2ps	%xmm2, %xmm2
.LVL1022:
.LBE7804:
.LBE7803:
.LBB7805:
.LBB7806:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
.LBE7806:
.LBE7805:
.LBB7808:
.LBB7809:
	.loc 4 931 0
	movups	48(%rax,%r9,4), %xmm9
.LVL1023:
.LBE7809:
.LBE7808:
.LBB7810:
.LBB7807:
	.loc 4 183 0
	addps	%xmm5, %xmm6
.LVL1024:
.LBE7807:
.LBE7810:
.LBB7811:
.LBB7812:
	.loc 3 767 0
	cvtdq2ps	%xmm8, %xmm5
.LVL1025:
.LBE7812:
.LBE7811:
.LBB7813:
.LBB7814:
	.loc 4 183 0
	mulps	%xmm1, %xmm5
.LBE7814:
.LBE7813:
.LBB7816:
.LBB7817:
	.loc 4 931 0
	movups	32(%rax,%r9,4), %xmm8
.LVL1026:
.LBE7817:
.LBE7816:
.LBB7818:
.LBB7819:
	.loc 4 183 0
	mulps	%xmm2, %xmm1
.LVL1027:
.LBE7819:
.LBE7818:
.LBB7821:
.LBB7822:
	.loc 4 931 0
	movups	(%rax,%r9,4), %xmm2
.LBE7822:
.LBE7821:
.LBB7823:
.LBB7815:
	.loc 4 183 0
	addps	%xmm5, %xmm7
.LVL1028:
.LBE7815:
.LBE7823:
.LBB7824:
.LBB7825:
	.loc 4 931 0
	movups	16(%rax,%r9,4), %xmm5
.LBE7825:
.LBE7824:
.LBB7826:
.LBB7820:
	.loc 4 183 0
	addps	%xmm1, %xmm0
.LVL1029:
.LBE7820:
.LBE7826:
.LBB7827:
.LBB7828:
.LBB7829:
.LBB7830:
	.loc 4 884 0
	movss	-36(%r11), %xmm1
.LVL1030:
.LBE7830:
.LBE7829:
.LBE7828:
.LBE7827:
.LBE7766:
.LBE7852:
	.loc 1 466 0
	cmpl	$4, %r8d
.LBB7853:
.LBB7849:
.LBB7831:
.LBB7832:
	.loc 4 743 0
	shufps	$0, %xmm1, %xmm1
.LVL1031:
.LBE7832:
.LBE7831:
.LBB7833:
.LBB7834:
	.loc 4 189 0
	mulps	%xmm1, %xmm9
.LVL1032:
.LBE7834:
.LBE7833:
.LBB7836:
.LBB7837:
	mulps	%xmm1, %xmm8
.LVL1033:
.LBE7837:
.LBE7836:
.LBB7839:
.LBB7840:
	mulps	%xmm1, %xmm5
.LVL1034:
.LBE7840:
.LBE7839:
.LBB7842:
.LBB7835:
	subps	%xmm9, %xmm3
.LBE7835:
.LBE7842:
.LBB7843:
.LBB7844:
	mulps	%xmm2, %xmm1
.LVL1035:
.LBE7844:
.LBE7843:
.LBB7846:
.LBB7838:
	subps	%xmm8, %xmm6
.LBE7838:
.LBE7846:
.LBB7847:
.LBB7841:
	subps	%xmm5, %xmm7
.LBE7841:
.LBE7847:
.LBB7848:
.LBB7845:
	subps	%xmm1, %xmm0
.LVL1036:
.LBE7845:
.LBE7848:
.LBE7849:
.LBE7853:
	.loc 1 466 0
	jne	.L516
.LVL1037:
.L428:
.LBE7854:
.LBB7855:
.LBB7856:
	.loc 4 980 0
	movups	%xmm0, (%rax)
.LVL1038:
.LBE7856:
.LBE7855:
.LBE7887:
	.loc 1 463 0
	addl	$16, %edi
.LVL1039:
	addq	$16, %rsi
.LVL1040:
	addq	$16, %rcx
.LVL1041:
.LBB7888:
.LBB7857:
.LBB7858:
	.loc 4 980 0
	movups	%xmm7, 16(%rax)
.LVL1042:
.LBE7858:
.LBE7857:
.LBE7888:
	.loc 1 463 0
	addq	$64, %rdx
.LVL1043:
	addq	$64, %rax
.LVL1044:
.LBB7889:
.LBB7859:
.LBB7860:
	.loc 4 980 0
	movups	%xmm6, -32(%rax)
.LVL1045:
.LBE7860:
.LBE7859:
.LBB7861:
.LBB7862:
	movups	%xmm3, -16(%rax)
.LVL1046:
.LBE7862:
.LBE7861:
.LBB7863:
.LBB7864:
	.loc 4 931 0
	movups	-16(%rdx), %xmm1
.LVL1047:
.LBE7864:
.LBE7863:
.LBB7865:
.LBB7866:
	movups	-32(%rdx), %xmm2
.LVL1048:
.LBE7866:
.LBE7865:
.LBB7867:
.LBB7868:
	.loc 4 980 0
	addps	%xmm1, %xmm3
.LBE7868:
.LBE7867:
.LBB7870:
.LBB7871:
	.loc 4 931 0
	movups	-48(%rdx), %xmm4
.LVL1049:
.LBE7871:
.LBE7870:
.LBB7872:
.LBB7873:
	.loc 4 980 0
	addps	%xmm2, %xmm6
.LBE7873:
.LBE7872:
.LBB7875:
.LBB7876:
	.loc 4 931 0
	movups	-64(%rdx), %xmm5
.LVL1050:
.LBE7876:
.LBE7875:
.LBB7877:
.LBB7878:
	.loc 4 980 0
	addps	%xmm4, %xmm7
.LBE7878:
.LBE7877:
.LBB7880:
.LBB7869:
	movups	%xmm3, -16(%rdx)
.LBE7869:
.LBE7880:
.LBB7881:
.LBB7882:
	addps	%xmm5, %xmm0
.LBE7882:
.LBE7881:
.LBB7884:
.LBB7874:
	movups	%xmm6, -32(%rdx)
.LBE7874:
.LBE7884:
.LBB7885:
.LBB7879:
	movups	%xmm7, -48(%rdx)
.LBE7879:
.LBE7885:
.LBB7886:
.LBB7883:
	movups	%xmm0, -64(%rdx)
.LVL1051:
.LBE7883:
.LBE7886:
.LBE7889:
	.loc 1 463 0
	cmpl	%edi, %r12d
	jle	.L414
	leal	(%rdi,%r10), %r9d
	.loc 1 463 0 is_stmt 0 discriminator 1
	movl	76(%r15), %r8d
.LVL1052:
	leal	-15(%r8), %r11d
	cmpl	%r9d, %r11d
	jle	.L415
	movl	72(%r15), %r9d
	jmp	.L416
.LVL1053:
	.p2align 4,,10
	.p2align 3
.L503:
	movq	-208(%rbp), %r15
.LVL1054:
.L414:
.LBE7718:
.LBE7711:
	.loc 1 457 0 is_stmt 1
	subl	$1, %ebx
.LVL1055:
	subq	$1, -200(%rbp)
	cmpl	$-1, %ebx
	je	.L417
	movl	72(%r15), %r9d
	movq	-200(%rbp), %r8
	leal	-1(%r9), %edx
.LVL1056:
	jmp	.L494
.LVL1057:
.L473:
.LBB7897:
.LBB7890:
	.loc 1 463 0
	movl	-172(%rbp), %r9d
.LBE7890:
	.loc 1 462 0
	xorl	%edi, %edi
.LVL1058:
	.p2align 4,,10
	.p2align 3
.L415:
.LBB7891:
	.loc 1 525 0
	cmpl	%r8d, %r9d
	jge	.L414
	movl	72(%r15), %r9d
	movq	%r15, -208(%rbp)
	movl	%r8d, -112(%rbp)
	leal	-1(%r9), %r11d
	leal	-2(%r9), %r13d
	movl	%r11d, -88(%rbp)
	leal	-3(%r9), %r11d
	movl	%r11d, -104(%rbp)
	leal	-4(%r9), %r11d
	movl	%r11d, -212(%rbp)
.LBB7892:
.LBB7893:
	.loc 1 528 0
	movslq	%r8d, %r11
	movl	-212(%rbp), %r15d
	leaq	0(,%r11,4), %r9
	movq	%r11, -120(%rbp)
	movq	%r9, -128(%rbp)
	leal	(%r8,%r8), %r9d
	movslq	%r9d, %r11
	addl	%r8d, %r9d
	movq	%r11, -136(%rbp)
	salq	$2, %r11
	movq	%r11, -144(%rbp)
	movslq	%r9d, %r11
	leaq	0(,%r11,4), %r9
	movq	%r11, -152(%rbp)
	leal	0(,%r8,4), %r11d
	movslq	%r11d, %r11
	movq	%r9, -160(%rbp)
	leaq	0(,%r11,4), %r9
	addq	%r11, %rsi
.LVL1059:
	movq	%r9, -168(%rbp)
	jmp	.L470
.LVL1060:
	.p2align 4,,10
	.p2align 3
.L517:
	movq	-120(%rbp), %r8
	movq	%rsi, %r9
	subq	%r11, %r9
	pxor	%xmm2, %xmm2
	movss	64(%r14), %xmm0
	.loc 1 527 0
	cmpl	%r13d, %ebx
	.loc 1 528 0
	movzbl	(%r9,%r8), %r9d
	movq	-128(%rbp), %r8
	movss	(%rax), %xmm3
	mulss	(%rax,%r8), %xmm0
	cvtsi2ss	%r9d, %xmm2
	mulss	96(%r14), %xmm2
	subss	%xmm0, %xmm2
	addss	%xmm2, %xmm3
	movss	%xmm3, (%rax)
.LVL1061:
	.loc 1 527 0
	jge	.L421
.L518:
	.loc 1 528 0
	movq	-136(%rbp), %r8
	movq	%rsi, %r9
	subq	%r11, %r9
	pxor	%xmm1, %xmm1
	movss	68(%r14), %xmm0
	.loc 1 527 0
	cmpl	-104(%rbp), %ebx
	.loc 1 528 0
	movzbl	(%r9,%r8), %r9d
	movq	-144(%rbp), %r8
	mulss	(%rax,%r8), %xmm0
	cvtsi2ss	%r9d, %xmm1
	mulss	100(%r14), %xmm1
	subss	%xmm0, %xmm1
	movaps	%xmm1, %xmm2
	addss	%xmm3, %xmm2
	movss	%xmm2, (%rax)
.LVL1062:
	.loc 1 527 0
	jge	.L423
.L519:
	.loc 1 528 0
	movq	-152(%rbp), %r8
	movq	%rsi, %r9
	subq	%r11, %r9
	pxor	%xmm0, %xmm0
	movss	72(%r14), %xmm1
	.loc 1 527 0
	cmpl	%r15d, %ebx
	.loc 1 528 0
	movzbl	(%r9,%r8), %r9d
	movq	-160(%rbp), %r8
	mulss	(%rax,%r8), %xmm1
	cvtsi2ss	%r9d, %xmm0
	mulss	104(%r14), %xmm0
	subss	%xmm1, %xmm0
	movaps	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	movss	%xmm1, (%rax)
.LVL1063:
	.loc 1 527 0
	jge	.L425
.L520:
	.loc 1 528 0
	movzbl	(%rsi), %r9d
	pxor	%xmm0, %xmm0
	movq	-168(%rbp), %r8
	movss	76(%r14), %xmm2
	cvtsi2ss	%r9d, %xmm0
	mulss	(%rax,%r8), %xmm2
	mulss	108(%r14), %xmm0
	subss	%xmm2, %xmm0
	addss	%xmm1, %xmm0
.L426:
	movss	%xmm0, (%rax)
.LVL1064:
.LBE7893:
.LBE7892:
	.loc 1 525 0
	addl	$1, %edi
.LVL1065:
	addq	$1, %rcx
.LVL1066:
	addq	$4, %rax
.LVL1067:
	addq	$4, %rdx
.LVL1068:
.LBB7895:
	.loc 1 533 0
	addss	-4(%rdx), %xmm0
	movss	%xmm0, -4(%rdx)
.LVL1069:
.LBE7895:
	.loc 1 525 0
	cmpl	%edi, %r12d
	jle	.L503
	addq	$1, %rsi
.LVL1070:
	.loc 1 525 0 is_stmt 0 discriminator 1
	leal	(%rdi,%r10), %r9d
	cmpl	-112(%rbp), %r9d
	jge	.L503
.LVL1071:
.L470:
.LBB7896:
.LBB7894:
	.loc 1 527 0 is_stmt 1 discriminator 1
	cmpl	-88(%rbp), %ebx
	jl	.L517
	.loc 1 530 0
	movzbl	(%rcx), %r9d
	pxor	%xmm0, %xmm0
	movss	96(%r14), %xmm2
	.loc 1 527 0
	cmpl	%r13d, %ebx
	.loc 1 530 0
	subss	64(%r14), %xmm2
	movss	(%rax), %xmm3
	cvtsi2ss	%r9d, %xmm0
	mulss	%xmm0, %xmm2
	addss	%xmm2, %xmm3
	movss	%xmm3, (%rax)
.LVL1072:
	.loc 1 527 0
	jl	.L518
.L421:
	.loc 1 530 0
	movzbl	(%rcx), %r9d
	pxor	%xmm0, %xmm0
	movss	100(%r14), %xmm1
	subss	68(%r14), %xmm1
	.loc 1 527 0
	cmpl	-104(%rbp), %ebx
	.loc 1 530 0
	cvtsi2ss	%r9d, %xmm0
	mulss	%xmm0, %xmm1
	movaps	%xmm1, %xmm2
	addss	%xmm3, %xmm2
	movss	%xmm2, (%rax)
.LVL1073:
	.loc 1 527 0
	jl	.L519
.L423:
	.loc 1 530 0
	movzbl	(%rcx), %r9d
	pxor	%xmm3, %xmm3
	movss	104(%r14), %xmm0
	.loc 1 527 0
	cmpl	%r15d, %ebx
	.loc 1 530 0
	subss	72(%r14), %xmm0
	cvtsi2ss	%r9d, %xmm3
	mulss	%xmm3, %xmm0
	movaps	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	movss	%xmm1, (%rax)
.LVL1074:
	.loc 1 527 0
	jl	.L520
.L425:
	.loc 1 530 0
	movzbl	(%rcx), %r9d
	pxor	%xmm2, %xmm2
	movss	108(%r14), %xmm0
	subss	76(%r14), %xmm0
	cvtsi2ss	%r9d, %xmm2
	mulss	%xmm2, %xmm0
	addss	%xmm1, %xmm0
	jmp	.L426
.LVL1075:
	.p2align 4,,10
	.p2align 3
.L508:
	movq	-112(%rbp), %r15
	movl	72(%r15), %r9d
.LVL1076:
	jmp	.L433
.LVL1077:
.L417:
	movl	-96(%rbp), %edi
	addl	%edi, -172(%rbp)
	movq	-248(%rbp), %rbx
.LVL1078:
	movl	-172(%rbp), %eax
.LVL1079:
	addq	%rbx, -192(%rbp)
	addl	%edi, -176(%rbp)
	movq	-240(%rbp), %rbx
	addq	%rbx, -184(%rbp)
	cmpl	%eax, -216(%rbp)
	jg	.L412
	leaq	-64(%rbp), %rsi
	leaq	-72(%rbp), %rdi
	movq	%r14, %r13
	movq	%r15, %r14
	call	GOMP_loop_dynamic_next
.LVL1080:
	testb	%al, %al
	jne	.L413
	movq	%r15, %r10
	movq	%r13, %r14
.LVL1081:
.L409:
	movq	%r10, -88(%rbp)
	call	GOMP_loop_end
.LVL1082:
.LBE7894:
.LBE7896:
.LBE7891:
.LBE7897:
.LBE7710:
.LBE7598:
.LBE7597:
.LBB7900:
	.loc 1 539 0
	movq	-88(%rbp), %r10
	movl	72(%r10), %eax
	movq	%r10, -96(%rbp)
	movl	%eax, %r15d
	movl	%eax, -172(%rbp)
	addl	$3, %eax
	testl	%r15d, %r15d
	cmovns	%r15d, %eax
	sarl	$2, %eax
	movl	%eax, %ebx
	call	omp_get_num_threads
.LVL1083:
	movl	%eax, %r12d
	call	omp_get_thread_num
.LVL1084:
	movl	%ebx, %edi
	imull	%eax, %edi
	leal	(%rdi,%rbx), %edx
	movl	%edi, %esi
	movl	%edi, -88(%rbp)
	cmpl	%r15d, %edx
	cmovg	%r15d, %edx
	cmpl	%r15d, %esi
	movl	%edx, -152(%rbp)
	jge	.L459
	movslq	%eax, %rdi
	addl	%r12d, %eax
	movq	-96(%rbp), %r10
	movq	%rdi, -160(%rbp)
	movl	%ebx, %edi
	imull	%r12d, %edi
	movq	%r10, %r15
	movl	%edi, -192(%rbp)
	movl	%eax, %edi
	addl	$1, %eax
	imull	%ebx, %edi
	imull	%ebx, %eax
	movl	%edi, -168(%rbp)
	subl	%edi, %eax
	movl	%eax, -184(%rbp)
.L463:
	movslq	-88(%rbp), %r8
.L462:
.LVL1085:
.LBB7901:
.LBB7902:
	.loc 1 543 0
	movl	76(%r15), %eax
	.loc 1 605 0
	movq	%rsp, -96(%rbp)
.LVL1086:
.LBB7903:
.LBB7904:
	.loc 2 430 0
	movq	%r8, %rdi
.LBE7904:
.LBE7903:
.LBB7909:
.LBB7910:
	movq	-160(%rbp), %r10
.LBE7910:
.LBE7909:
.LBB7912:
.LBB7913:
	movq	%r8, %r13
.LBE7913:
.LBE7912:
.LBB7918:
.LBB7919:
	.loc 5 90 0
	xorl	%esi, %esi
.LBE7919:
.LBE7918:
.LBB7922:
.LBB7914:
	.loc 2 430 0
	movq	%r8, -144(%rbp)
.LBE7914:
.LBE7922:
	.loc 1 543 0
	addl	$24, %eax
	cltq
	leaq	18(,%rax,4), %rax
	andq	$-16, %rax
	subq	%rax, %rsp
	.loc 1 545 0
	movq	48(%r15), %rax
	.loc 1 543 0
	leaq	3(%rsp), %r9
.LBB7923:
.LBB7905:
	.loc 2 430 0
	movq	72(%rax), %rdx
.LBE7905:
.LBE7923:
	.loc 1 543 0
	shrq	$2, %r9
	leaq	0(,%r9,4), %r12
.LVL1087:
.LBB7924:
.LBB7925:
	.loc 5 53 0
	movq	%r9, -136(%rbp)
.LBE7925:
.LBE7924:
.LBB7929:
.LBB7906:
	.loc 2 430 0
	imulq	(%rdx), %rdi
.LBE7906:
.LBE7929:
	.loc 1 552 0
	addq	$48, %r12
.LVL1088:
.LBB7930:
.LBB7907:
	.loc 2 430 0
	movq	%rdi, %rdx
	addq	16(%rax), %rdx
.LBE7907:
.LBE7930:
	.loc 1 546 0
	movq	56(%r15), %rax
.LVL1089:
.LBB7931:
.LBB7908:
	.loc 2 430 0
	movq	%rdx, -104(%rbp)
.LBE7908:
.LBE7931:
.LBB7932:
.LBB7911:
	movq	72(%rax), %rdx
	imulq	(%rdx), %r10
	addq	16(%rax), %r10
.LVL1090:
.LBE7911:
.LBE7932:
	.loc 1 547 0
	movq	40(%r15), %rax
.LVL1091:
.LBB7933:
.LBB7915:
	.loc 2 430 0
	movq	72(%rax), %rdx
.LBE7915:
.LBE7933:
	.loc 1 551 0
	leaq	48(%r10), %rbx
	movq	%r10, -128(%rbp)
.LBB7934:
.LBB7916:
	.loc 2 430 0
	imulq	(%rdx), %r13
.LBE7916:
.LBE7934:
.LBB7935:
.LBB7926:
	.loc 5 53 0
	movl	8(,%r9,4), %edx
.LBE7926:
.LBE7935:
.LBB7936:
.LBB7920:
	.loc 5 90 0
	movq	%rbx, %rdi
.LBE7920:
.LBE7936:
.LBB7937:
.LBB7917:
	.loc 2 430 0
	addq	16(%rax), %r13
.LVL1092:
.LBE7917:
.LBE7937:
.LBB7938:
.LBB7927:
	.loc 5 53 0
	movq	0(,%r9,4), %rax
.LVL1093:
	movl	%edx, 8(%r10)
.LVL1094:
.LBE7927:
.LBE7938:
.LBB7939:
.LBB7940:
	movl	%edx, 20(%r10)
.LVL1095:
	movq	%rax, 12(%r10)
.LBE7940:
.LBE7939:
.LBB7941:
.LBB7928:
	movq	%rax, (%r10)
.LBE7928:
.LBE7941:
.LBB7942:
.LBB7943:
	movq	%rax, 24(%r10)
	movq	8(%r10), %rax
	movq	%rax, 32(%r10)
	movq	16(%r10), %rax
	movq	%rax, 40(%r10)
.LVL1096:
.LBE7943:
.LBE7942:
.LBB7944:
.LBB7921:
	.loc 5 90 0
	movslq	76(%r15), %rdx
	salq	$2, %rdx
.LVL1097:
	call	memset
.LVL1098:
.LBE7921:
.LBE7944:
	.loc 1 552 0
	movslq	76(%r15), %rcx
.LBB7945:
.LBB7946:
	.loc 5 53 0
	movq	-104(%rbp), %rsi
	movq	%r12, %rdi
	leaq	0(,%rcx,4), %rdx
.LBE7946:
.LBE7945:
	.loc 1 552 0
	movl	%ecx, -120(%rbp)
.LVL1099:
.LBB7948:
.LBB7947:
	.loc 5 53 0
	movq	%rcx, -112(%rbp)
	call	memcpy
.LVL1100:
.LBE7947:
.LBE7948:
.LBB7949:
.LBB7950:
	movq	-128(%rbp), %r10
	movq	-136(%rbp), %r9
.LBE7950:
.LBE7949:
.LBB7953:
	.loc 1 556 0
	movl	-120(%rbp), %r11d
	movq	-112(%rbp), %rcx
	movq	-144(%rbp), %r8
.LBE7953:
.LBB7986:
.LBB7951:
	.loc 5 53 0
	movq	(%r10), %rax
.LBE7951:
.LBE7986:
.LBB7987:
	.loc 1 556 0
	testl	%r11d, %r11d
.LBE7987:
.LBB7988:
.LBB7952:
	.loc 5 53 0
	movq	%rax, 0(,%r9,4)
	movq	8(%r10), %rax
	movq	%rax, 8(,%r9,4)
	movq	16(%r10), %rax
	movq	%rax, 16(,%r9,4)
	movq	24(%r10), %rax
	movq	%rax, 24(,%r9,4)
	movq	32(%r10), %rax
	movq	%rax, 32(,%r9,4)
	movq	40(%r10), %rax
	movq	%rax, 40(,%r9,4)
.LVL1101:
.LBE7952:
.LBE7988:
.LBB7989:
	.loc 1 556 0
	jle	.L521
	movss	36(,%r9,4), %xmm1
	.loc 1 556 0 is_stmt 0 discriminator 2
	xorl	%esi, %esi
	movss	40(,%r9,4), %xmm7
	movss	44(,%r9,4), %xmm6
	movss	12(,%r9,4), %xmm0
	movss	16(,%r9,4), %xmm5
	movss	20(,%r9,4), %xmm4
.LVL1102:
	.p2align 4,,10
	.p2align 3
.L468:
.LBB7954:
	.loc 1 567 0 is_stmt 1 discriminator 2
	movss	-24(%r12), %xmm2
.LBB7955:
.LBB7956:
	.loc 4 946 0 discriminator 2
	movaps	%xmm2, %xmm8
.LBE7956:
.LBE7955:
	.loc 1 567 0 discriminator 2
	movss	(%r12), %xmm3
.LBB7959:
.LBB7957:
	.loc 4 946 0 discriminator 2
	unpcklps	%xmm0, %xmm8
	movaps	%xmm3, %xmm0
	unpcklps	%xmm1, %xmm0
.LBE7957:
.LBE7959:
.LBB7960:
.LBB7961:
	movss	-12(%rbx), %xmm1
	insertps	$0x10, -24(%rbx), %xmm1
.LBE7961:
.LBE7960:
.LBB7964:
.LBB7958:
	movlhps	%xmm8, %xmm0
.LBE7958:
.LBE7964:
.LBB7965:
.LBB7962:
	movss	-36(%rbx), %xmm8
	insertps	$0x10, -48(%rbx), %xmm8
.LBE7962:
.LBE7965:
.LBB7966:
.LBB7967:
	.loc 4 189 0 discriminator 2
	mulps	32(%r14), %xmm0
.LBE7967:
.LBE7966:
.LBB7969:
.LBB7963:
	.loc 4 946 0 discriminator 2
	movlhps	%xmm8, %xmm1
.LBE7963:
.LBE7969:
.LBB7970:
.LBB7968:
	.loc 4 189 0 discriminator 2
	mulps	16(%r14), %xmm1
	subps	%xmm1, %xmm0
	movaps	%xmm7, %xmm1
	movaps	%xmm6, %xmm7
	movaps	%xmm3, %xmm6
.LBE7968:
.LBE7970:
.LBB7971:
.LBB7972:
	.loc 7 58 0 discriminator 2
	haddps	%xmm0, %xmm0
.LBE7972:
.LBE7971:
.LBB7973:
.LBB7974:
	haddps	%xmm0, %xmm0
.LBE7974:
.LBE7973:
.LBB7975:
.LBB7976:
	.loc 4 960 0 discriminator 2
	movss	%xmm0, (%rbx)
.LBE7976:
.LBE7975:
.LBB7977:
.LBB7978:
.LBB7979:
.LBB7980:
.LBB7981:
.LBB7982:
	.loc 3 63 0 discriminator 2
	cvtss2sd	%xmm0, %xmm0
.LBE7982:
.LBE7981:
.LBB7983:
.LBB7984:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm0, %edx
	movaps	%xmm5, %xmm0
	movaps	%xmm4, %xmm5
	movaps	%xmm2, %xmm4
.LBE7984:
.LBE7983:
.LBE7980:
.LBE7979:
.LBE7978:
.LBE7977:
	.loc 1 575 0 discriminator 2
	testl	%edx, %edx
	setg	%al
	negl	%eax
	cmpl	$255, %edx
	cmovbe	%edx, %eax
.LBE7954:
	.loc 1 556 0 discriminator 2
	addl	$1, %esi
.LVL1103:
	addq	$4, %r12
.LVL1104:
.LBB7985:
	.loc 1 575 0 discriminator 2
	movb	%al, 0(%r13)
.LBE7985:
	.loc 1 556 0 discriminator 2
	movslq	76(%r15), %rax
	addq	$4, %rbx
.LVL1105:
	addq	$1, %r13
.LVL1106:
	cmpl	%esi, %eax
	jg	.L468
.LVL1107:
.L469:
.LBE7989:
	.loc 1 577 0
	movq	40(%r15), %rdx
.LVL1108:
	movq	%r8, %rdi
	movq	%r8, -104(%rbp)
.LBB7990:
.LBB7991:
	.loc 2 430 0
	movq	72(%rdx), %rsi
.LBE7991:
.LBE7990:
	.loc 1 577 0
	imulq	(%rsi), %rdi
.LBB7992:
.LBB7993:
	.loc 5 90 0
	xorl	%esi, %esi
.LBE7993:
.LBE7992:
	.loc 1 577 0
	leaq	-1(%rax,%rdi), %r13
.LVL1109:
.LBB7996:
.LBB7997:
	.loc 5 53 0
	movq	-12(%r12), %rax
.LBE7997:
.LBE7996:
	.loc 1 577 0
	addq	16(%rdx), %r13
.LVL1110:
.LBB8000:
.LBB7998:
	.loc 5 53 0
	movl	-4(%r12), %edx
.LVL1111:
.LBE7998:
.LBE8000:
.LBB8001:
.LBB7994:
	.loc 5 90 0
	movq	%rbx, %rdi
.LBE7994:
.LBE8001:
	.loc 1 584 0
	subq	$4, %rbx
.LVL1112:
.LBB8002:
.LBB8003:
	.loc 5 53 0
	movq	%rax, 16(%rbx)
.LVL1113:
.LBE8003:
.LBE8002:
.LBB8005:
.LBB7999:
	movq	%rax, 4(%rbx)
	movl	%edx, 12(%rbx)
.LVL1114:
.LBE7999:
.LBE8005:
.LBB8006:
.LBB8007:
	movq	%rax, 28(%rbx)
	movq	12(%rbx), %rax
.LBE8007:
.LBE8006:
.LBB8009:
.LBB8004:
	movl	%edx, 24(%rbx)
.LVL1115:
.LBE8004:
.LBE8009:
.LBB8010:
.LBB8008:
	movq	%rax, 36(%rbx)
	movq	20(%rbx), %rax
	movq	%rax, 44(%rbx)
.LVL1116:
.LBE8008:
.LBE8010:
	.loc 1 581 0
	movslq	76(%r15), %rdx
	salq	$2, %rdx
.LVL1117:
.LBB8011:
.LBB7995:
	.loc 5 90 0
	subq	%rdx, %rdi
.LVL1118:
	call	memset
.LVL1119:
.LBE7995:
.LBE8011:
.LBB8012:
.LBB8013:
	.loc 5 53 0
	movq	4(%rbx), %rax
.LBE8013:
.LBE8012:
.LBB8017:
	.loc 1 585 0
	movl	76(%r15), %edx
	movq	-104(%rbp), %r8
.LBE8017:
.LBB8208:
.LBB8014:
	.loc 5 53 0
	movq	%rax, (%r12)
	movq	12(%rbx), %rax
.LBE8014:
.LBE8208:
.LBB8209:
	.loc 1 585 0
	movl	%edx, %esi
.LBE8209:
.LBB8210:
.LBB8015:
	.loc 5 53 0
	movq	%rax, 8(%r12)
	movq	20(%rbx), %rax
	movq	%rax, 16(%r12)
	movq	28(%rbx), %rax
	movq	%rax, 24(%r12)
	movq	36(%rbx), %rax
	movq	%rax, 32(%r12)
	movq	44(%rbx), %rax
.LBE8015:
.LBE8210:
.LBB8211:
	.loc 1 585 0
	subl	$1, %esi
.LBE8211:
.LBB8212:
.LBB8016:
	.loc 5 53 0
	movq	%rax, 40(%r12)
.LVL1120:
.LBE8016:
.LBE8212:
	.loc 1 583 0
	leaq	-4(%r12), %rax
.LVL1121:
.LBB8213:
	.loc 1 585 0
	js	.L466
	cmpl	$4, %edx
	jle	.L475
	subl	$5, %edx
	movq	$-20, %rdi
	shrl	$2, %edx
	movss	20(%r12), %xmm5
	salq	$4, %rdx
	movss	16(%r12), %xmm4
	subq	%rdx, %rdi
	movss	12(%r12), %xmm7
	movss	44(%r12), %xmm3
	movss	40(%r12), %xmm2
	movss	36(%r12), %xmm6
	addq	%rdi, %r12
.LVL1122:
	.p2align 4,,10
	.p2align 3
.L465:
.LBB8018:
	.loc 1 596 0 discriminator 2
	movss	12(%rax), %xmm8
	movss	36(%rax), %xmm9
.LBB8019:
.LBB8020:
	.loc 4 946 0 discriminator 2
	movaps	%xmm8, %xmm0
	movaps	%xmm9, %xmm1
	unpcklps	%xmm5, %xmm0
.LBE8020:
.LBE8019:
	.loc 1 596 0 discriminator 2
	movss	8(%rax), %xmm5
.LBB8033:
.LBB8021:
	.loc 4 946 0 discriminator 2
	unpcklps	%xmm3, %xmm1
.LBE8021:
.LBE8033:
.LBB8034:
.LBB8035:
	movss	36(%rbx), %xmm3
	insertps	$0x10, 48(%rbx), %xmm3
.LBE8035:
.LBE8034:
.LBB8049:
.LBB8022:
	movlhps	%xmm1, %xmm0
.LBE8022:
.LBE8049:
.LBB8050:
.LBB8036:
	movss	12(%rbx), %xmm1
	insertps	$0x10, 24(%rbx), %xmm1
.LBE8036:
.LBE8050:
.LBB8051:
.LBB8052:
	.loc 4 189 0 discriminator 2
	mulps	48(%r14), %xmm0
.LBE8052:
.LBE8051:
.LBB8062:
.LBB8037:
	.loc 4 946 0 discriminator 2
	movlhps	%xmm3, %xmm1
.LBE8037:
.LBE8062:
	.loc 1 596 0 discriminator 2
	movss	32(%rax), %xmm3
.LBB8063:
.LBB8053:
	.loc 4 189 0 discriminator 2
	mulps	16(%r14), %xmm1
	subps	%xmm1, %xmm0
.LBE8053:
.LBE8063:
.LBB8064:
.LBB8065:
.LBB8066:
.LBB8067:
.LBB8068:
.LBB8069:
	.loc 3 63 0 discriminator 2
	pxor	%xmm1, %xmm1
.LBE8069:
.LBE8068:
.LBE8067:
.LBE8066:
.LBE8065:
.LBE8064:
.LBB8135:
.LBB8136:
	.loc 7 58 0 discriminator 2
	haddps	%xmm0, %xmm0
.LBE8136:
.LBE8135:
.LBB8141:
.LBB8142:
	haddps	%xmm0, %xmm0
.LBE8142:
.LBE8141:
.LBB8147:
.LBB8148:
	.loc 4 960 0 discriminator 2
	movss	%xmm0, (%rbx)
.LBE8148:
.LBE8147:
.LBB8153:
.LBB8124:
.LBB8113:
.LBB8102:
.LBB8081:
.LBB8070:
	.loc 3 63 0 discriminator 2
	movzbl	0(%r13), %edx
	cvtsi2ss	%edx, %xmm1
	addss	%xmm1, %xmm0
.LBE8070:
.LBE8081:
.LBE8102:
.LBE8113:
.LBE8124:
.LBE8153:
.LBB8154:
.LBB8023:
	.loc 4 946 0 discriminator 2
	movaps	%xmm3, %xmm1
	unpcklps	%xmm2, %xmm1
.LBE8023:
.LBE8154:
.LBB8155:
.LBB8125:
.LBB8114:
.LBB8103:
.LBB8082:
.LBB8071:
	.loc 3 63 0 discriminator 2
	cvtss2sd	%xmm0, %xmm0
.LBE8071:
.LBE8082:
.LBB8083:
.LBB8084:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm0, %ecx
.LBE8084:
.LBE8083:
.LBE8103:
.LBE8114:
.LBE8125:
.LBE8155:
.LBB8156:
.LBB8024:
	.loc 4 946 0 discriminator 2
	movaps	%xmm5, %xmm0
	unpcklps	%xmm4, %xmm0
.LBE8024:
.LBE8156:
	.loc 1 596 0 discriminator 2
	movss	4(%rax), %xmm4
.LBB8157:
.LBB8025:
	.loc 4 946 0 discriminator 2
	movlhps	%xmm1, %xmm0
.LBE8025:
.LBE8157:
	.loc 1 604 0 discriminator 2
	testl	%ecx, %ecx
	setg	%dl
	negl	%edx
	cmpl	$255, %ecx
	cmovbe	%ecx, %edx
	movb	%dl, 0(%r13)
.LBB8158:
.LBB8038:
	.loc 4 946 0 discriminator 2
	movss	32(%rbx), %xmm2
	movss	8(%rbx), %xmm1
	insertps	$0x10, 44(%rbx), %xmm2
	insertps	$0x10, 20(%rbx), %xmm1
.LBE8038:
.LBE8158:
.LBB8159:
.LBB8054:
	.loc 4 189 0 discriminator 2
	mulps	48(%r14), %xmm0
.LBE8054:
.LBE8159:
.LBB8160:
.LBB8039:
	.loc 4 946 0 discriminator 2
	movlhps	%xmm2, %xmm1
.LBE8039:
.LBE8160:
	.loc 1 596 0 discriminator 2
	movss	28(%rax), %xmm2
.LBB8161:
.LBB8055:
	.loc 4 189 0 discriminator 2
	mulps	16(%r14), %xmm1
	subps	%xmm1, %xmm0
.LBE8055:
.LBE8161:
.LBB8162:
.LBB8126:
.LBB8115:
.LBB8104:
.LBB8089:
.LBB8072:
	.loc 3 63 0 discriminator 2
	pxor	%xmm1, %xmm1
.LBE8072:
.LBE8089:
.LBE8104:
.LBE8115:
.LBE8126:
.LBE8162:
.LBB8163:
.LBB8137:
	.loc 7 58 0 discriminator 2
	haddps	%xmm0, %xmm0
.LBE8137:
.LBE8163:
.LBB8164:
.LBB8143:
	haddps	%xmm0, %xmm0
.LBE8143:
.LBE8164:
.LBB8165:
.LBB8149:
	.loc 4 960 0 discriminator 2
	movss	%xmm0, -4(%rbx)
.LBE8149:
.LBE8165:
.LBB8166:
.LBB8127:
.LBB8116:
.LBB8105:
.LBB8090:
.LBB8073:
	.loc 3 63 0 discriminator 2
	movzbl	-1(%r13), %edx
	cvtsi2ss	%edx, %xmm1
	addss	%xmm1, %xmm0
.LBE8073:
.LBE8090:
.LBE8105:
.LBE8116:
.LBE8127:
.LBE8166:
.LBB8167:
.LBB8026:
	.loc 4 946 0 discriminator 2
	movaps	%xmm2, %xmm1
	unpcklps	%xmm6, %xmm1
.LBE8026:
.LBE8167:
.LBB8168:
.LBB8128:
.LBB8117:
.LBB8106:
.LBB8091:
.LBB8074:
	.loc 3 63 0 discriminator 2
	cvtss2sd	%xmm0, %xmm0
.LBE8074:
.LBE8091:
.LBB8092:
.LBB8085:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm0, %ecx
.LBE8085:
.LBE8092:
.LBE8106:
.LBE8117:
.LBE8128:
.LBE8168:
.LBB8169:
.LBB8027:
	.loc 4 946 0 discriminator 2
	movaps	%xmm4, %xmm0
	unpcklps	%xmm7, %xmm0
	movlhps	%xmm1, %xmm0
.LBE8027:
.LBE8169:
	.loc 1 604 0 discriminator 2
	testl	%ecx, %ecx
	setg	%dl
	negl	%edx
	cmpl	$255, %ecx
	cmovbe	%ecx, %edx
	movb	%dl, -1(%r13)
.LBB8170:
.LBB8040:
	.loc 4 946 0 discriminator 2
	movss	28(%rbx), %xmm6
	movss	4(%rbx), %xmm1
	insertps	$0x10, 40(%rbx), %xmm6
	insertps	$0x10, 16(%rbx), %xmm1
.LBE8040:
.LBE8170:
.LBB8171:
.LBB8056:
	.loc 4 189 0 discriminator 2
	mulps	48(%r14), %xmm0
.LBE8056:
.LBE8171:
.LBB8172:
.LBB8041:
	.loc 4 946 0 discriminator 2
	movlhps	%xmm6, %xmm1
.LBE8041:
.LBE8172:
.LBB8173:
.LBB8057:
	.loc 4 189 0 discriminator 2
	mulps	16(%r14), %xmm1
	subps	%xmm1, %xmm0
.LBE8057:
.LBE8173:
.LBB8174:
.LBB8129:
.LBB8118:
.LBB8107:
.LBB8093:
.LBB8075:
	.loc 3 63 0 discriminator 2
	pxor	%xmm1, %xmm1
.LBE8075:
.LBE8093:
.LBE8107:
.LBE8118:
.LBE8129:
.LBE8174:
.LBB8175:
.LBB8138:
	.loc 7 58 0 discriminator 2
	haddps	%xmm0, %xmm0
.LBE8138:
.LBE8175:
.LBB8176:
.LBB8144:
	haddps	%xmm0, %xmm0
.LBE8144:
.LBE8176:
.LBB8177:
.LBB8150:
	.loc 4 960 0 discriminator 2
	movss	%xmm0, -8(%rbx)
.LBE8150:
.LBE8177:
.LBB8178:
.LBB8130:
.LBB8119:
.LBB8108:
.LBB8094:
.LBB8076:
	.loc 3 63 0 discriminator 2
	movzbl	-2(%r13), %edx
	cvtsi2ss	%edx, %xmm1
	addss	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
.LBE8076:
.LBE8094:
.LBB8095:
.LBB8086:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm0, %ecx
.LBE8086:
.LBE8095:
.LBE8108:
.LBE8119:
.LBE8130:
.LBE8178:
	.loc 1 604 0 discriminator 2
	testl	%ecx, %ecx
	setg	%dl
	negl	%edx
	cmpl	$255, %ecx
	cmovbe	%ecx, %edx
	movb	%dl, -2(%r13)
	.loc 1 596 0 discriminator 2
	movss	(%rax), %xmm7
	movss	24(%rax), %xmm6
.LBB8179:
.LBB8028:
	.loc 4 946 0 discriminator 2
	movaps	%xmm6, %xmm1
	movaps	%xmm7, %xmm0
	unpcklps	%xmm9, %xmm1
	unpcklps	%xmm8, %xmm0
.LBE8028:
.LBE8179:
.LBB8180:
.LBB8042:
	movss	24(%rbx), %xmm8
	insertps	$0x10, 36(%rbx), %xmm8
.LBE8042:
.LBE8180:
.LBB8181:
.LBB8029:
	movlhps	%xmm1, %xmm0
.LBE8029:
.LBE8181:
.LBB8182:
.LBB8043:
	movss	(%rbx), %xmm1
	insertps	$0x10, 12(%rbx), %xmm1
.LBE8043:
.LBE8182:
.LBB8183:
.LBB8058:
	.loc 4 189 0 discriminator 2
	mulps	48(%r14), %xmm0
.LBE8058:
.LBE8183:
.LBB8184:
.LBB8044:
	.loc 4 946 0 discriminator 2
	movlhps	%xmm8, %xmm1
.LBE8044:
.LBE8184:
.LBB8185:
.LBB8059:
	.loc 4 189 0 discriminator 2
	mulps	16(%r14), %xmm1
	subps	%xmm1, %xmm0
.LBE8059:
.LBE8185:
.LBB8186:
.LBB8131:
.LBB8120:
.LBB8109:
.LBB8096:
.LBB8077:
	.loc 3 63 0 discriminator 2
	pxor	%xmm1, %xmm1
.LBE8077:
.LBE8096:
.LBE8109:
.LBE8120:
.LBE8131:
.LBE8186:
.LBB8187:
.LBB8139:
	.loc 7 58 0 discriminator 2
	haddps	%xmm0, %xmm0
.LBE8139:
.LBE8187:
.LBB8188:
.LBB8145:
	haddps	%xmm0, %xmm0
.LBE8145:
.LBE8188:
.LBB8189:
.LBB8151:
	.loc 4 960 0 discriminator 2
	movss	%xmm0, -12(%rbx)
.LBE8151:
.LBE8189:
.LBB8190:
.LBB8132:
.LBB8121:
.LBB8110:
.LBB8097:
.LBB8078:
	.loc 3 63 0 discriminator 2
	movzbl	-3(%r13), %edx
	cvtsi2ss	%edx, %xmm1
	addss	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
.LBE8078:
.LBE8097:
.LBB8098:
.LBB8087:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm0, %ecx
.LBE8087:
.LBE8098:
.LBE8110:
.LBE8121:
.LBE8132:
.LBE8190:
	.loc 1 604 0 discriminator 2
	testl	%ecx, %ecx
	setg	%dl
	negl	%edx
	cmpl	$255, %ecx
	cmovbe	%ecx, %edx
.LBE8018:
	.loc 1 585 0 discriminator 2
	subq	$16, %rax
	subl	$4, %esi
.LVL1123:
.LBB8205:
	.loc 1 604 0 discriminator 2
	movb	%dl, -3(%r13)
.LBE8205:
	.loc 1 585 0 discriminator 2
	subq	$16, %rbx
.LVL1124:
	subq	$4, %r13
.LVL1125:
	cmpq	%r12, %rax
	jne	.L465
.LVL1126:
	.p2align 4,,10
	.p2align 3
.L467:
.LBB8206:
.LBB8191:
.LBB8030:
	.loc 4 946 0
	movss	36(%r12), %xmm1
	movss	12(%r12), %xmm0
	insertps	$0x10, 48(%r12), %xmm1
.LBE8030:
.LBE8191:
.LBB8192:
.LBB8045:
	movss	36(%rbx), %xmm2
.LBE8045:
.LBE8192:
.LBB8193:
.LBB8031:
	insertps	$0x10, 24(%r12), %xmm0
.LBE8031:
.LBE8193:
.LBB8194:
.LBB8046:
	insertps	$0x10, 48(%rbx), %xmm2
.LBE8046:
.LBE8194:
.LBB8195:
.LBB8032:
	movlhps	%xmm1, %xmm0
.LBE8032:
.LBE8195:
.LBB8196:
.LBB8047:
	movss	12(%rbx), %xmm1
	insertps	$0x10, 24(%rbx), %xmm1
.LBE8047:
.LBE8196:
.LBB8197:
.LBB8060:
	.loc 4 189 0
	mulps	48(%r14), %xmm0
.LBE8060:
.LBE8197:
.LBB8198:
.LBB8048:
	.loc 4 946 0
	movlhps	%xmm2, %xmm1
.LBE8048:
.LBE8198:
.LBB8199:
.LBB8061:
	.loc 4 189 0
	mulps	16(%r14), %xmm1
	subps	%xmm1, %xmm0
.LBE8061:
.LBE8199:
.LBB8200:
.LBB8133:
.LBB8122:
.LBB8111:
.LBB8099:
.LBB8079:
	.loc 3 63 0
	pxor	%xmm1, %xmm1
.LBE8079:
.LBE8099:
.LBE8111:
.LBE8122:
.LBE8133:
.LBE8200:
.LBB8201:
.LBB8140:
	.loc 7 58 0
	haddps	%xmm0, %xmm0
.LBE8140:
.LBE8201:
.LBB8202:
.LBB8146:
	haddps	%xmm0, %xmm0
.LBE8146:
.LBE8202:
.LBB8203:
.LBB8152:
	.loc 4 960 0
	movss	%xmm0, (%rbx)
.LBE8152:
.LBE8203:
.LBB8204:
.LBB8134:
.LBB8123:
.LBB8112:
.LBB8100:
.LBB8080:
	.loc 3 63 0
	movzbl	0(%r13), %eax
	cvtsi2ss	%eax, %xmm1
	addss	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
.LBE8080:
.LBE8100:
.LBB8101:
.LBB8088:
	.loc 3 827 0
	cvtsd2si	%xmm0, %edx
.LBE8088:
.LBE8101:
.LBE8112:
.LBE8123:
.LBE8134:
.LBE8204:
	.loc 1 604 0
	testl	%edx, %edx
	setg	%al
	negl	%eax
	cmpl	$255, %edx
	cmovbe	%edx, %eax
.LBE8206:
	.loc 1 585 0
	subl	$1, %esi
.LVL1127:
	subq	$4, %r12
.LVL1128:
.LBB8207:
	.loc 1 604 0
	movb	%al, 0(%r13)
.LBE8207:
	.loc 1 585 0
	subq	$4, %rbx
.LVL1129:
	subq	$1, %r13
.LVL1130:
	cmpl	$-1, %esi
	jne	.L467
.LVL1131:
.L466:
	addl	$1, -88(%rbp)
.LVL1132:
	addq	$1, %r8
.LBE8213:
	movq	-96(%rbp), %rsp
	movl	-88(%rbp), %eax
.LVL1133:
	cmpl	%eax, -152(%rbp)
	jg	.L462
	movl	-168(%rbp), %edi
	movl	-172(%rbp), %ebx
.LVL1134:
	movl	%edi, %eax
.LVL1135:
	addl	-184(%rbp), %eax
	movl	%edi, -88(%rbp)
.LVL1136:
	cmpl	%ebx, %eax
	cmovg	%ebx, %eax
	movl	%eax, -152(%rbp)
	movl	%edi, %eax
	movl	-192(%rbp), %edi
	addl	%edi, %eax
	movl	%eax, -168(%rbp)
	subl	%edi, %eax
	cmpl	%eax, %ebx
	jg	.L463
.LVL1137:
.L459:
	call	GOMP_barrier
.LVL1138:
.LBE7902:
.LBE7901:
.LBE7900:
	.loc 1 392 0
	movq	-56(%rbp), %rax
	xorq	%fs:40, %rax
	jne	.L522
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.LVL1139:
	.p2align 4,,10
	.p2align 3
.L512:
	.cfi_restore_state
.LBB8218:
.LBB7899:
.LBB7898:
.LBB7709:
.LBB7708:
	.loc 1 405 0
	movl	-172(%rbp), %ecx
.LBE7708:
	.loc 1 404 0
	xorl	%r9d, %r9d
	jmp	.L434
.LVL1140:
.L475:
.LBE7709:
.LBE7898:
.LBE7899:
.LBE8218:
.LBB8219:
.LBB8217:
.LBB8216:
.LBB8214:
	.loc 1 585 0
	movq	%rax, %r12
	jmp	.L467
.LVL1141:
.L521:
.LBE8214:
.LBB8215:
	.loc 1 556 0
	movq	%rcx, %rax
	jmp	.L469
.LVL1142:
.L522:
.LBE8215:
.LBE8216:
.LBE8217:
.LBE8219:
	.loc 1 392 0
	call	__stack_chk_fail
.LVL1143:
	.cfi_endproc
.LFE12404:
	.size	_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_._omp_fn.5, .-_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_._omp_fn.5
	.section	.text.unlikely._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_._omp_fn.5,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_,comdat
.LCOLDE5:
	.section	.text._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_._omp_fn.5,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_,comdat
.LHOTE5:
	.section	.text.unlikely,"ax",@progbits
.LCOLDB7:
	.text
.LHOTB7:
	.p2align 4,,15
	.type	_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.6, @function
_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.6:
.LFB12405:
	.loc 1 855 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
.LVL1144:
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
.LBB8220:
	.loc 1 855 0
	movl	32(%rdi), %r14d
.LBE8220:
	movq	8(%rdi), %rbx
.LVL1145:
.LBB8250:
	movq	%rdi, 8(%rsp)
	leal	3(%r14), %eax
	testl	%r14d, %r14d
	cmovns	%r14d, %eax
	sarl	$2, %eax
	movl	%eax, 4(%rsp)
	call	omp_get_num_threads
.LVL1146:
	movl	%eax, %r13d
	call	omp_get_thread_num
.LVL1147:
	movl	4(%rsp), %ecx
	movl	%ecx, %ebp
	imull	%eax, %ebp
	leal	(%rcx,%rbp), %r12d
	cmpl	%r14d, %r12d
	cmovg	%r14d, %r12d
	cmpl	%ebp, %r14d
	jle	.L523
	addl	%r13d, %eax
	movl	%ecx, %r15d
	movq	8(%rsp), %rdi
	imull	%r13d, %r15d
	movl	%eax, %r13d
	addl	$1, %eax
	imull	%ecx, %r13d
	movsd	.LC6(%rip), %xmm1
	movl	36(%rdi), %edx
	imull	%ecx, %eax
	subl	%r13d, %eax
	movl	%eax, 4(%rsp)
	.p2align 4,,10
	.p2align 3
.L526:
.LVL1148:
.LBB8221:
.LBB8222:
	.loc 1 858 0
	movl	%ebp, %r9d
	movq	(%rdi), %rax
.LBB8223:
	.loc 1 861 0
	xorl	%ecx, %ecx
.LBE8223:
	.loc 1 858 0
	imull	%edx, %r9d
.LBB8248:
	.loc 1 861 0
	xorl	%esi, %esi
.LBE8248:
	.loc 1 858 0
	movslq	%r9d, %r9
	movq	%r9, %r8
	addq	16(%rax), %r8
.LVL1149:
	movq	16(%rdi), %rax
	.loc 1 859 0
	movq	%r9, %r10
	addq	16(%rax), %r10
.LVL1150:
	movq	24(%rdi), %rax
	.loc 1 860 0
	addq	16(%rax), %r9
.LVL1151:
.LBB8249:
	.loc 1 861 0
	testl	%edx, %edx
	jle	.L528
.LVL1152:
	.p2align 4,,10
	.p2align 3
.L532:
	movzbl	(%r8,%rcx), %edx
	movzbl	40(%rdi), %eax
	subl	%edx, %eax
	cltd
	xorl	%edx, %eax
	subl	%edx, %eax
	cltq
	leaq	(%rbx,%rax,8), %rdx
.LBB8224:
.LBB8225:
.LBB8226:
.LBB8227:
.LBB8228:
.LBB8229:
.LBB8230:
	.loc 3 63 0 discriminator 2
	movsd	32(%rdx), %xmm0
	mulsd	%xmm1, %xmm0
.LBE8230:
.LBE8229:
.LBB8231:
.LBB8232:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm0, %r11d
.LBE8232:
.LBE8231:
.LBE8228:
.LBE8227:
.LBE8226:
.LBE8225:
.LBB8233:
.LBB8234:
.LBB8235:
.LBB8236:
.LBB8237:
.LBB8238:
	.loc 3 63 0 discriminator 2
	pxor	%xmm0, %xmm0
.LBE8238:
.LBE8237:
.LBE8236:
.LBE8235:
.LBE8234:
.LBE8233:
	.loc 1 864 0 discriminator 2
	testl	%r11d, %r11d
	setg	%al
	negl	%eax
	cmpl	$255, %r11d
	cmovbe	%r11d, %eax
	movb	%al, (%r10,%rcx)
.LBB8246:
.LBB8245:
.LBB8244:
.LBB8243:
.LBB8240:
.LBB8239:
	.loc 3 63 0 discriminator 2
	movzbl	(%r8,%rcx), %eax
	cvtsi2sd	%eax, %xmm0
	mulsd	32(%rdx), %xmm0
.LBE8239:
.LBE8240:
.LBB8241:
.LBB8242:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm0, %edx
.LBE8242:
.LBE8241:
.LBE8243:
.LBE8244:
.LBE8245:
.LBE8246:
	.loc 1 865 0 discriminator 2
	testl	%edx, %edx
	setg	%al
	negl	%eax
	cmpl	$255, %edx
	cmovbe	%edx, %eax
.LBE8224:
	.loc 1 861 0 discriminator 2
	addl	$1, %esi
.LVL1153:
.LBB8247:
	.loc 1 865 0 discriminator 2
	movb	%al, (%r9,%rcx)
.LBE8247:
	.loc 1 861 0 discriminator 2
	movl	36(%rdi), %edx
	addq	$1, %rcx
.LVL1154:
	cmpl	%esi, %edx
	jg	.L532
.LVL1155:
.L528:
	addl	$1, %ebp
.LVL1156:
	cmpl	%r12d, %ebp
	jl	.L526
	movl	4(%rsp), %eax
	movl	%r13d, %ebp
.LVL1157:
	leal	0(%r13,%rax), %r12d
	cmpl	%r14d, %r12d
	cmovg	%r14d, %r12d
	addl	%r15d, %r13d
	movl	%r13d, %eax
	subl	%r15d, %eax
	cmpl	%eax, %r14d
	jg	.L526
.LVL1158:
.L523:
.LBE8249:
.LBE8222:
.LBE8221:
.LBE8250:
	.loc 1 855 0
	addq	$24, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
.LVL1159:
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE12405:
	.size	_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.6, .-_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.6
	.section	.text.unlikely
.LCOLDE7:
	.text
.LHOTE7:
	.section	.text.unlikely
.LCOLDB8:
	.text
.LHOTB8:
	.p2align 4,,15
	.type	_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.7, @function
_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.7:
.LFB12406:
	.loc 1 873 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
.LVL1160:
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	movq	%rdi, %r15
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$8, %rsp
	.cfi_def_cfa_offset 64
.LBB8251:
	.loc 1 873 0
	movl	24(%rdi), %ebp
	leal	3(%rbp), %r12d
	testl	%ebp, %ebp
	cmovns	%ebp, %r12d
	sarl	$2, %r12d
	call	omp_get_num_threads
.LVL1161:
	movl	%eax, %ebx
	call	omp_get_thread_num
.LVL1162:
	movl	%r12d, %r10d
	imull	%eax, %r10d
	leal	(%r12,%r10), %r11d
	cmpl	%ebp, %r11d
	cmovg	%ebp, %r11d
	cmpl	%r10d, %ebp
	jle	.L535
	movl	%r12d, %r13d
	addl	%ebx, %eax
	movl	28(%r15), %r14d
	imull	%ebx, %r13d
	movl	%eax, %ebx
	addl	$1, %eax
	imull	%r12d, %ebx
	movsd	.LC6(%rip), %xmm2
	imull	%eax, %r12d
	subl	%ebx, %r12d
	.p2align 4,,10
	.p2align 3
.L538:
.LVL1163:
.LBB8252:
.LBB8253:
	.loc 1 876 0
	movl	%r10d, %esi
	movq	(%r15), %rax
	movq	16(%r15), %rdx
	imull	%r14d, %esi
.LBB8254:
	.loc 1 879 0
	xorl	%ecx, %ecx
.LBE8254:
	.loc 1 876 0
	movslq	%esi, %rsi
	movq	%rsi, %r9
	addq	16(%rax), %r9
.LVL1164:
	movq	8(%r15), %rax
	.loc 1 877 0
	movq	%rsi, %r8
	addq	16(%rax), %r8
.LVL1165:
	movslq	32(%r15), %rax
	leaq	(%rax,%rax,2), %rax
	salq	$5, %rax
	addq	(%rdx), %rax
.LBB8263:
	.loc 1 879 0
	xorl	%edx, %edx
.LVL1166:
.LBE8263:
	.loc 1 878 0
	addq	16(%rax), %rsi
.LVL1167:
.LBB8264:
	.loc 1 879 0
	testl	%r14d, %r14d
	jle	.L540
.LVL1168:
	.p2align 4,,10
	.p2align 3
.L544:
.LBB8255:
.LBB8256:
.LBB8257:
.LBB8258:
.LBB8259:
.LBB8260:
	.loc 3 63 0 discriminator 2
	movzbl	(%r8,%rdx), %eax
	pxor	%xmm0, %xmm0
	pxor	%xmm1, %xmm1
	cvtsi2sd	%eax, %xmm0
	movzbl	(%r9,%rdx), %eax
	cvtsi2sd	%eax, %xmm1
	mulsd	%xmm2, %xmm0
	divsd	%xmm1, %xmm0
.LBE8260:
.LBE8259:
.LBB8261:
.LBB8262:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm0, %eax
.LBE8262:
.LBE8261:
.LBE8258:
.LBE8257:
.LBE8256:
.LBE8255:
	.loc 1 880 0 discriminator 2
	testl	%eax, %eax
	setg	%r14b
	negl	%r14d
	cmpl	$255, %eax
	cmova	%r14d, %eax
	.loc 1 879 0 discriminator 2
	addl	$1, %ecx
.LVL1169:
	.loc 1 880 0 discriminator 2
	movb	%al, (%rsi,%rdx)
	.loc 1 879 0 discriminator 2
	movl	28(%r15), %r14d
	addq	$1, %rdx
.LVL1170:
	cmpl	%ecx, %r14d
	jg	.L544
.LVL1171:
.L540:
	addl	$1, %r10d
.LVL1172:
	cmpl	%r11d, %r10d
	jl	.L538
	leal	(%rbx,%r12), %r11d
	movl	%ebx, %r10d
.LVL1173:
	cmpl	%ebp, %r11d
	cmovg	%ebp, %r11d
	addl	%r13d, %ebx
	movl	%ebx, %eax
	subl	%r13d, %eax
	cmpl	%eax, %ebp
	jg	.L538
.L535:
.LBE8264:
.LBE8253:
.LBE8252:
.LBE8251:
	.loc 1 873 0
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
.LVL1174:
	ret
	.cfi_endproc
.LFE12406:
	.size	_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.7, .-_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.7
	.section	.text.unlikely
.LCOLDE8:
	.text
.LHOTE8:
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC9:
	.string	"ERROR: lapack routine LAPACKE_zgetrf() for solving a of simga = %f failed!\n"
	.section	.text.unlikely
	.align 2
.LCOLDB10:
.LHOTB10:
	.align 2
	.type	_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_.part.37, @function
_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_.part.37:
.LFB12436:
	.loc 1 624 0
	.cfi_startproc
.LVL1175:
	pushq	%rax
	.cfi_def_cfa_offset 16
.LBB8265:
.LBB8266:
	.file 8 "/usr/include/x86_64-linux-gnu/bits/stdio2.h"
	.loc 8 98 0
	movq	stderr(%rip), %rdi
	movl	$.LC9, %edx
	movl	$1, %esi
	movb	$1, %al
	call	__fprintf_chk
.LVL1176:
.LBE8266:
.LBE8265:
	.loc 1 657 0
	orl	$-1, %edi
	call	exit
.LVL1177:
	.cfi_endproc
.LFE12436:
	.size	_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_.part.37, .-_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_.part.37
.LCOLDE10:
.LHOTE10:
	.section	.rodata.str1.8
	.align 8
.LC11:
	.string	"basic_string::_M_construct null not valid"
	.section	.text.unlikely
	.align 2
.LCOLDB12:
	.text
.LHOTB12:
	.align 2
	.p2align 4,,15
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_.isra.64, @function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_.isra.64:
.LFB12463:
	.file 9 "/usr/include/c++/5/bits/basic_string.h"
	.loc 9 454 0
	.cfi_startproc
.LVL1178:
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
.LBB8305:
.LBB8306:
.LBB8307:
	.loc 9 141 0
	leaq	16(%rdi), %r13
.LBE8307:
.LBE8306:
.LBE8305:
	.loc 9 454 0
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	subq	$24, %rsp
	.cfi_def_cfa_offset 64
.LBB8361:
.LBB8308:
.LBB8309:
	.loc 9 109 0
	movq	%r13, (%rdi)
.LBE8309:
.LBE8308:
.LBE8361:
	.loc 9 454 0
	movq	%fs:40, %rax
	movq	%rax, 8(%rsp)
	xorl	%eax, %eax
.LVL1179:
.LBB8362:
	.loc 9 456 0
	testq	%rsi, %rsi
	je	.L549
	movq	%rdi, %rbx
.LBB8310:
.LBB8311:
	.file 10 "/usr/include/c++/5/bits/char_traits.h"
	.loc 10 267 0
	movq	%rsi, %rdi
.LVL1180:
	movq	%rsi, %r12
.LVL1181:
	call	strlen
.LVL1182:
.LBE8311:
.LBE8310:
.LBB8313:
.LBB8314:
.LBB8315:
.LBB8316:
.LBB8317:
.LBB8318:
	.file 11 "/usr/include/c++/5/bits/basic_string.tcc"
	.loc 11 221 0
	cmpq	$15, %rax
.LBE8318:
.LBE8317:
.LBE8316:
.LBE8315:
.LBE8314:
.LBE8313:
.LBB8358:
.LBB8312:
	.loc 10 267 0
	movq	%rax, %rbp
.LVL1183:
.LBE8312:
.LBE8358:
.LBB8359:
.LBB8356:
.LBB8354:
.LBB8352:
.LBB8350:
.LBB8348:
	.loc 11 219 0
	movq	%rax, (%rsp)
	.loc 11 221 0
	ja	.L561
.LVL1184:
.LBB8319:
.LBB8320:
.LBB8321:
	.loc 9 296 0
	cmpq	$1, %rax
	je	.L562
.LVL1185:
.LBB8322:
.LBB8323:
	.loc 10 288 0
	testq	%rax, %rax
	jne	.L563
.LVL1186:
.L553:
.LBE8323:
.LBE8322:
.LBE8321:
.LBE8320:
.LBE8319:
	.loc 11 236 0
	movq	(%rsp), %rax
.LVL1187:
.LBB8334:
.LBB8335:
.LBB8336:
.LBB8337:
	.loc 10 243 0
	movq	(%rbx), %rdx
.LBE8337:
.LBE8336:
.LBB8339:
.LBB8340:
	.loc 9 131 0
	movq	%rax, 8(%rbx)
.LVL1188:
.LBE8340:
.LBE8339:
.LBB8341:
.LBB8338:
	.loc 10 243 0
	movb	$0, (%rdx,%rax)
.LVL1189:
.LBE8338:
.LBE8341:
.LBE8335:
.LBE8334:
.LBE8348:
.LBE8350:
.LBE8352:
.LBE8354:
.LBE8356:
.LBE8359:
.LBE8362:
	.loc 9 456 0
	movq	8(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L564
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
.LVL1190:
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
.LVL1191:
	popq	%r13
	.cfi_def_cfa_offset 8
.LVL1192:
	ret
.LVL1193:
.L561:
	.cfi_restore_state
.LBB8363:
.LBB8360:
.LBB8357:
.LBB8355:
.LBB8353:
.LBB8351:
.LBB8349:
	.loc 11 223 0
	movq	%rbx, %rdi
	xorl	%edx, %edx
	movq	%rsp, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm
.LVL1194:
	movq	%rax, %rdi
.LVL1195:
.LBB8342:
.LBB8343:
	.loc 9 127 0
	movq	%rax, (%rbx)
.LVL1196:
.LBE8343:
.LBE8342:
.LBB8344:
.LBB8345:
	.loc 9 159 0
	movq	(%rsp), %rax
	movq	%rax, 16(%rbx)
.LVL1197:
.L551:
.LBE8345:
.LBE8344:
.LBB8346:
.LBB8332:
.LBB8330:
.LBB8326:
.LBB8324:
	.loc 10 290 0
	movq	%rbp, %rdx
	movq	%r12, %rsi
	call	memcpy
.LVL1198:
	jmp	.L553
.LVL1199:
.L562:
	movzbl	(%r12), %eax
.LVL1200:
.LBE8324:
.LBE8326:
.LBB8327:
.LBB8328:
	.loc 10 243 0
	movb	%al, 16(%rbx)
	jmp	.L553
.LVL1201:
.L549:
.LBE8328:
.LBE8327:
.LBE8330:
.LBE8332:
.LBE8346:
	.loc 11 216 0
	movl	$.LC11, %edi
.LVL1202:
	call	_ZSt19__throw_logic_errorPKc
.LVL1203:
.L563:
.LBB8347:
.LBB8333:
.LBB8331:
.LBB8329:
.LBB8325:
	.loc 10 288 0
	movq	%r13, %rdi
	jmp	.L551
.LVL1204:
.L564:
.LBE8325:
.LBE8329:
.LBE8331:
.LBE8333:
.LBE8347:
.LBE8349:
.LBE8351:
.LBE8353:
.LBE8355:
.LBE8357:
.LBE8360:
.LBE8363:
	.loc 9 456 0
	call	__stack_chk_fail
.LVL1205:
	.cfi_endproc
.LFE12463:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_.isra.64, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_.isra.64
	.section	.text.unlikely
.LCOLDE12:
	.text
.LHOTE12:
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC13:
	.string	"\n"
	.section	.text.unlikely
.LCOLDB14:
	.text
.LHOTB14:
	.p2align 4,,15
	.type	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.constprop.80, @function
_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.constprop.80:
.LFB12480:
	.file 12 "/usr/include/c++/5/ostream"
	.loc 12 556 0
	.cfi_startproc
.LVL1206:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	.loc 12 561 0
	movl	$1, %edx
	.loc 12 556 0
	movq	%rdi, %rbx
	.loc 12 561 0
	movl	$.LC13, %esi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.LVL1207:
	.loc 12 564 0
	movq	%rbx, %rax
	popq	%rbx
	.cfi_def_cfa_offset 8
.LVL1208:
	ret
	.cfi_endproc
.LFE12480:
	.size	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.constprop.80, .-_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.constprop.80
	.section	.text.unlikely
.LCOLDE14:
	.text
.LHOTE14:
	.section	.rodata.str1.1
.LC17:
	.string	"\t"
	.section	.text.unlikely
.LCOLDB18:
	.text
.LHOTB18:
	.p2align 4,,15
	.type	_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.8, @function
_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.8:
.LFB12407:
	.loc 1 885 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA12407
.LVL1209:
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rdi, %r13
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$88, %rsp
	.cfi_def_cfa_offset 144
.LBB8405:
	.loc 1 885 0
	movl	36(%rdi), %eax
.LBE8405:
	movq	16(%rdi), %r14
.LVL1210:
.LBB8458:
	movl	%eax, %r12d
	movl	%eax, 68(%rsp)
	addl	$3, %eax
	testl	%r12d, %r12d
	cmovns	%r12d, %eax
	sarl	$2, %eax
	movl	%eax, %ebx
	movl	%ebx, %r15d
	call	omp_get_num_threads
.LVL1211:
	movl	%eax, %ebp
	call	omp_get_thread_num
.LVL1212:
	imull	%eax, %r15d
	leal	(%rbx,%r15), %edx
	cmpl	%r12d, %edx
	cmovg	%r12d, %edx
	cmpl	%r15d, %r12d
	movl	%edx, 60(%rsp)
	jle	.L567
	movl	%ebx, %edi
	addl	%ebp, %eax
	movl	40(%r13), %edx
	imull	%ebp, %edi
	pxor	%xmm3, %xmm3
	movsd	.LC15(%rip), %xmm4
	movq	%r14, %r8
	movl	%edi, 72(%rsp)
	movl	%eax, %edi
	addl	$1, %eax
	imull	%ebx, %edi
	imull	%ebx, %eax
	movl	%edi, 64(%rsp)
	subl	%edi, %eax
	movl	%eax, 76(%rsp)
.LVL1213:
	.p2align 4,,10
	.p2align 3
.L570:
.LBB8406:
.LBB8407:
	.loc 1 888 0
	movl	%r15d, %r14d
	movq	0(%r13), %rax
.LBB8408:
	.loc 1 890 0
	xorl	%r12d, %r12d
.LBE8408:
	.loc 1 888 0
	imull	%edx, %r14d
	movslq	%r14d, %r14
	movq	%r14, %rcx
	addq	16(%rax), %rcx
.LVL1214:
	movq	8(%r13), %rax
	.loc 1 889 0
	addq	16(%rax), %r14
.LVL1215:
.LBB8457:
	.loc 1 890 0
	testl	%edx, %edx
	jle	.L579
	movq	%r14, %rax
	movq	%r8, %r14
.LVL1216:
	movq	%rax, %r8
.LVL1217:
	jmp	.L585
.LVL1218:
	.p2align 4,,10
	.p2align 3
.L581:
	movq	24(%r13), %rax
	movslq	%ebx, %rbx
	movslq	%ebp, %rbp
	leaq	(%rbx,%rbx,2), %rdx
.LBB8409:
.LBB8410:
.LBB8411:
.LBB8412:
.LBB8413:
.LBB8414:
.LBB8415:
	.loc 3 63 0 discriminator 2
	pxor	%xmm0, %xmm0
	movq	(%rax), %rsi
.LVL1219:
.LBE8415:
.LBE8414:
.LBE8413:
.LBE8412:
.LBE8411:
.LBE8410:
	.loc 1 899 0 discriminator 2
	movl	40(%r13), %eax
.LVL1220:
	salq	$5, %rdx
	imull	%r15d, %eax
	movq	16(%rsi,%rdx), %rdx
	cltq
	addq	%r12, %rax
.LVL1221:
.LBB8430:
.LBB8428:
.LBB8424:
.LBB8422:
.LBB8418:
.LBB8416:
	.loc 3 63 0 discriminator 2
	movzbl	(%rdx,%rax), %edx
	cvtsi2sd	%edx, %xmm0
	leaq	0(%rbp,%rbp,2), %rdx
	salq	$5, %rdx
.LBE8416:
.LBE8418:
.LBE8422:
.LBE8424:
.LBE8428:
.LBE8430:
	.loc 1 900 0 discriminator 2
	movq	16(%rsi,%rdx), %rdx
.LBB8431:
.LBB8429:
.LBB8425:
.LBB8423:
.LBB8419:
.LBB8417:
	.loc 3 63 0 discriminator 2
	movzbl	(%rdx,%rax), %eax
	mulsd	%xmm0, %xmm2
	pxor	%xmm0, %xmm0
	cvtsi2sd	%eax, %xmm0
	mulsd	%xmm0, %xmm1
	addsd	%xmm2, %xmm1
.LBE8417:
.LBE8419:
.LBB8420:
.LBB8421:
	.loc 3 827 0 discriminator 2
	cvtsd2si	%xmm1, %eax
.LVL1222:
.LBE8421:
.LBE8420:
.LBE8423:
.LBE8425:
.LBB8426:
.LBB8427:
	.file 13 "/home/xiaocen/Software/opencv/include/opencv2/core/operations.hpp"
	.loc 13 132 0 discriminator 2
	cmpl	$255, %eax
	movl	%eax, %edx
	jbe	.L578
	.loc 13 132 0 is_stmt 0
	testl	%eax, %eax
	setg	%dl
	negl	%edx
.L578:
.LVL1223:
.LBE8427:
.LBE8426:
.LBE8429:
.LBE8431:
	.loc 1 901 0 is_stmt 1
	movb	%dl, (%r8,%r12)
.LVL1224:
.LBE8409:
	.loc 1 890 0
	movl	40(%r13), %edx
	leal	1(%r12), %eax
.LVL1225:
	addq	$1, %r12
.LVL1226:
	cmpl	%eax, %edx
	jle	.L589
.LVL1227:
.L585:
.LBB8456:
	.loc 1 891 0
	movzbl	(%rcx,%r12), %eax
	pxor	%xmm1, %xmm1
	movsd	(%r14), %xmm2
	cvtsi2sd	%eax, %xmm1
	.loc 1 892 0
	movl	32(%r13), %eax
	subl	$1, %eax
	.loc 1 891 0
	movapd	%xmm1, %xmm0
	divsd	%xmm2, %xmm0
	cvttsd2si	%xmm0, %ebx
.LVL1228:
.LBB8432:
.LBB8433:
.LBB8434:
.LBB8435:
.LBB8436:
.LBB8437:
	.loc 3 63 0
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ebx, %xmm0
.LBE8437:
.LBE8436:
.LBE8435:
.LBE8434:
.LBE8433:
.LBE8432:
	leal	1(%rbx), %ebp
	cmpl	%eax, %ebx
	cmovge	%ebx, %ebp
.LVL1229:
.LBB8447:
.LBB8446:
.LBB8443:
.LBB8442:
.LBB8439:
.LBB8438:
	mulsd	%xmm2, %xmm0
.LBE8438:
.LBE8439:
.LBB8440:
.LBB8441:
	.loc 3 827 0
	cvtsd2si	%xmm0, %edx
.LVL1230:
.LBE8441:
.LBE8440:
.LBE8442:
.LBE8443:
.LBB8444:
.LBB8445:
	.loc 13 132 0
	cmpl	$255, %edx
	movl	%edx, %eax
	jbe	.L574
	testl	%edx, %edx
	setg	%al
	negl	%eax
.L574:
.LVL1231:
.LBE8445:
.LBE8444:
.LBE8446:
.LBE8447:
	.loc 1 896 0
	pxor	%xmm0, %xmm0
	movzbl	%al, %eax
	cvtsi2sd	%eax, %xmm0
	subsd	%xmm0, %xmm1
	divsd	%xmm2, %xmm1
.LVL1232:
	.loc 1 897 0
	movapd	%xmm4, %xmm2
	subsd	%xmm1, %xmm2
.LVL1233:
	.loc 1 898 0
	ucomisd	%xmm2, %xmm3
	ja	.L575
	.loc 1 898 0 is_stmt 0 discriminator 2
	ucomisd	%xmm1, %xmm3
	jbe	.L581
.L575:
.LBB8448:
.LBB8449:
	.loc 12 221 0 is_stmt 1
	movapd	%xmm2, %xmm0
	movl	$_ZSt4cout, %edi
	movq	%r8, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movsd	%xmm3, 32(%rsp)
	movsd	%xmm4, 24(%rsp)
	movsd	%xmm2, 16(%rsp)
	movsd	%xmm1, (%rsp)
.LVL1234:
	call	_ZNSo9_M_insertIdEERSoT_
.LVL1235:
.LBE8449:
.LBE8448:
.LBB8450:
.LBB8451:
	.loc 12 561 0
	movl	$1, %edx
	movl	$.LC17, %esi
	movq	%rax, %rdi
	movq	%rax, 8(%rsp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.LVL1236:
.LBE8451:
.LBE8450:
.LBB8452:
.LBB8453:
	.loc 12 221 0
	movsd	(%rsp), %xmm1
	movq	8(%rsp), %r9
	movapd	%xmm1, %xmm0
	movq	%r9, %rdi
	call	_ZNSo9_M_insertIdEERSoT_
.LVL1237:
.LBE8453:
.LBE8452:
.LBB8454:
.LBB8455:
	.loc 12 561 0
	movl	$1, %edx
	movl	$.LC13, %esi
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.LVL1238:
	movsd	(%rsp), %xmm1
	movq	40(%rsp), %rcx
	movsd	16(%rsp), %xmm2
	movq	48(%rsp), %r8
	movsd	24(%rsp), %xmm4
	movsd	32(%rsp), %xmm3
	jmp	.L581
.LVL1239:
	.p2align 4,,10
	.p2align 3
.L589:
	movq	%r14, %r8
.LVL1240:
.L579:
	addl	$1, %r15d
.LVL1241:
	cmpl	60(%rsp), %r15d
	jl	.L570
	movl	64(%rsp), %edi
	movl	76(%rsp), %eax
	movl	68(%rsp), %ecx
	addl	%edi, %eax
	movl	%edi, %r15d
.LVL1242:
	cmpl	%ecx, %eax
	cmovg	%ecx, %eax
	movl	%eax, 60(%rsp)
	movl	%edi, %eax
	movl	72(%rsp), %edi
	addl	%edi, %eax
	movl	%eax, 64(%rsp)
	subl	%edi, %eax
	cmpl	%eax, %ecx
	jg	.L570
.LVL1243:
.L567:
.LBE8455:
.LBE8454:
.LBE8456:
.LBE8457:
.LBE8407:
.LBE8406:
.LBE8458:
	.loc 1 885 0
	addq	$88, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
.LVL1244:
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE12407:
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
.LLSDA12407:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE12407-.LLSDACSB12407
.LLSDACSB12407:
.LLSDACSE12407:
	.text
	.size	_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.8, .-_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.8
	.section	.text.unlikely
.LCOLDE18:
	.text
.LHOTE18:
	.section	.text.unlikely._ZN2cv3MatC2ENS_5Size_IiEEi,"axG",@progbits,_ZN2cv3MatC5ENS_5Size_IiEEi,comdat
	.align 2
.LCOLDB19:
	.section	.text._ZN2cv3MatC2ENS_5Size_IiEEi,"axG",@progbits,_ZN2cv3MatC5ENS_5Size_IiEEi,comdat
.LHOTB19:
	.align 2
	.p2align 4,,15
	.weak	_ZN2cv3MatC2ENS_5Size_IiEEi
	.type	_ZN2cv3MatC2ENS_5Size_IiEEi, @function
_ZN2cv3MatC2ENS_5Size_IiEEi:
.LFB9420:
	.loc 2 85 0
	.cfi_startproc
.LVL1245:
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
.LBB8477:
.LBB8478:
.LBB8479:
	.loc 2 352 0
	movl	4(%rsi), %ecx
	.loc 2 353 0
	andl	$4095, %edx
.LVL1246:
.LBE8479:
.LBE8478:
.LBE8477:
	.loc 2 85 0
	movq	%fs:40, %rax
	movq	%rax, 8(%rsp)
	xorl	%eax, %eax
.LVL1247:
.LBB8502:
	leaq	8(%rdi), %rax
.LVL1248:
.LBB8485:
.LBB8486:
	.loc 2 738 0
	movq	$0, 88(%rdi)
	movq	$0, 80(%rdi)
.LBE8486:
.LBE8485:
.LBB8489:
.LBB8480:
	.loc 2 352 0
	movl	%ecx, (%rsp)
.LBE8480:
.LBE8489:
.LBB8490:
.LBB8491:
	.loc 2 60 0
	movl	$1124007936, (%rdi)
.LBE8491:
.LBE8490:
.LBB8494:
.LBB8481:
	.loc 2 353 0
	movl	%edx, %ecx
.LBE8481:
.LBE8494:
	.loc 2 85 0
	movq	%rax, 64(%rdi)
.LVL1249:
.LBB8495:
.LBB8487:
	.loc 2 738 0
	leaq	80(%rdi), %rax
.LBE8487:
.LBE8495:
.LBB8496:
.LBB8492:
	.loc 2 61 0
	movl	$0, 12(%rdi)
	movl	$0, 8(%rdi)
	movl	$0, 4(%rdi)
.LBE8492:
.LBE8496:
.LBB8497:
.LBB8482:
	.loc 2 353 0
	movq	%rsp, %rdx
.LBE8482:
.LBE8497:
.LBB8498:
.LBB8488:
	.loc 2 738 0
	movq	%rax, 72(%rdi)
.LBE8488:
.LBE8498:
	.loc 2 88 0
	movl	(%rsi), %eax
.LBB8499:
.LBB8483:
	.loc 2 353 0
	movl	$2, %esi
.LVL1250:
.LBE8483:
.LBE8499:
.LBB8500:
.LBB8493:
	.loc 2 62 0
	movq	$0, 48(%rdi)
	movq	$0, 40(%rdi)
	movq	$0, 32(%rdi)
	movq	$0, 16(%rdi)
	.loc 2 63 0
	movq	$0, 24(%rdi)
	.loc 2 64 0
	movq	$0, 56(%rdi)
.LVL1251:
.LBE8493:
.LBE8500:
.LBB8501:
.LBB8484:
	.loc 2 352 0
	movl	%eax, 4(%rsp)
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL1252:
.LBE8484:
.LBE8501:
.LBE8502:
	.loc 2 89 0
	movq	8(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L593
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L593:
	.cfi_restore_state
	call	__stack_chk_fail
.LVL1253:
	.cfi_endproc
.LFE9420:
	.size	_ZN2cv3MatC2ENS_5Size_IiEEi, .-_ZN2cv3MatC2ENS_5Size_IiEEi
	.section	.text.unlikely._ZN2cv3MatC2ENS_5Size_IiEEi,"axG",@progbits,_ZN2cv3MatC5ENS_5Size_IiEEi,comdat
.LCOLDE19:
	.section	.text._ZN2cv3MatC2ENS_5Size_IiEEi,"axG",@progbits,_ZN2cv3MatC5ENS_5Size_IiEEi,comdat
.LHOTE19:
	.weak	_ZN2cv3MatC1ENS_5Size_IiEEi
	.set	_ZN2cv3MatC1ENS_5Size_IiEEi,_ZN2cv3MatC2ENS_5Size_IiEEi
	.section	.text.unlikely._ZN2cv3MatD2Ev,"axG",@progbits,_ZN2cv3MatD5Ev,comdat
	.align 2
.LCOLDB20:
	.section	.text._ZN2cv3MatD2Ev,"axG",@progbits,_ZN2cv3MatD5Ev,comdat
.LHOTB20:
	.align 2
	.p2align 4,,15
	.weak	_ZN2cv3MatD2Ev
	.type	_ZN2cv3MatD2Ev, @function
_ZN2cv3MatD2Ev:
.LFB9447:
	.loc 2 274 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA9447
.LVL1254:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
.LBB8508:
.LBB8509:
.LBB8510:
	.loc 2 366 0
	movq	24(%rdi), %rax
.LBE8510:
.LBE8509:
.LBE8508:
	.loc 2 274 0
	movq	%rdi, %rbx
.LBB8517:
.LBB8515:
.LBB8513:
	.loc 2 366 0
	testq	%rax, %rax
	je	.L602
	lock subl	$1, (%rax)
	jne	.L602
	.loc 2 367 0
	call	_ZN2cv3Mat10deallocateEv
.LVL1255:
.L602:
.LBB8511:
	.loc 2 369 0
	movl	4(%rbx), %eax
.LBE8511:
	.loc 2 368 0
	movq	$0, 48(%rbx)
	movq	$0, 40(%rbx)
	movq	$0, 32(%rbx)
	movq	$0, 16(%rbx)
.LVL1256:
.LBB8512:
	.loc 2 369 0
	testl	%eax, %eax
	jle	.L600
	movq	64(%rbx), %rdx
	xorl	%eax, %eax
.LVL1257:
	.p2align 4,,10
	.p2align 3
.L601:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL1258:
	addq	$4, %rdx
	cmpl	%eax, 4(%rbx)
	jg	.L601
.LVL1259:
.L600:
.LBE8512:
.LBE8513:
.LBE8515:
	.loc 2 277 0
	movq	72(%rbx), %rdi
.LBB8516:
.LBB8514:
	.loc 2 371 0
	movq	$0, 24(%rbx)
.LVL1260:
.LBE8514:
.LBE8516:
	.loc 2 277 0
	addq	$80, %rbx
.LVL1261:
	cmpq	%rbx, %rdi
	je	.L609
.LBE8517:
	.loc 2 279 0
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 8
.LVL1262:
.LBB8518:
	.loc 2 278 0
	jmp	_ZN2cv8fastFreeEPv
.LVL1263:
	.p2align 4,,10
	.p2align 3
.L609:
	.cfi_restore_state
.LBE8518:
	.loc 2 279 0
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE9447:
	.section	.gcc_except_table
.LLSDA9447:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE9447-.LLSDACSB9447
.LLSDACSB9447:
.LLSDACSE9447:
	.section	.text._ZN2cv3MatD2Ev,"axG",@progbits,_ZN2cv3MatD5Ev,comdat
	.size	_ZN2cv3MatD2Ev, .-_ZN2cv3MatD2Ev
	.section	.text.unlikely._ZN2cv3MatD2Ev,"axG",@progbits,_ZN2cv3MatD5Ev,comdat
.LCOLDE20:
	.section	.text._ZN2cv3MatD2Ev,"axG",@progbits,_ZN2cv3MatD5Ev,comdat
.LHOTE20:
	.weak	_ZN2cv3MatD1Ev
	.set	_ZN2cv3MatD1Ev,_ZN2cv3MatD2Ev
	.section	.rodata.str1.8
	.align 8
.LC35:
	.string	"ERROR: lapack routine LAPACKE_zgetrs() for solving a of sigma = %f failed!\n"
	.align 8
.LC36:
	.string	"ERROR: lapack routine LAPACKE_zgetrf() for solving b of simga = %f failed!\n"
	.align 8
.LC37:
	.string	"ERROR: lapack routine LAPACKE_zgetrs() for solving b of sigma = %f failed!\n"
	.section	.text.unlikely._ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_,"axG",@progbits,_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_,comdat
	.align 2
.LCOLDB40:
	.section	.text._ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_,"axG",@progbits,_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_,comdat
.LHOTB40:
	.align 2
	.p2align 4,,15
	.weak	_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_
	.type	_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_, @function
_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_:
.LFB11284:
	.loc 1 625 0
	.cfi_startproc
.LVL1264:
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	movq	%rdx, %r15
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	.loc 1 626 0
	xorl	%ebx, %ebx
	.loc 1 625 0
	movapd	%xmm0, %xmm6
.LBB8645:
	.loc 1 641 0
	xorl	%ebp, %ebp
.LBE8645:
	.loc 1 625 0
	subq	$616, %rsp
	.cfi_def_cfa_offset 672
.LBB8757:
	.loc 1 633 0
	movsd	.LC21(%rip), %xmm2
.LBE8757:
	.loc 1 625 0
	movq	%rcx, 120(%rsp)
	leaq	144(%rsp), %r13
.LBB8806:
.LBB8758:
.LBB8759:
.LBB8760:
	.file 14 "/usr/include/c++/5/complex"
	.loc 14 750 0
	movsd	.LC22(%rip), %xmm1
.LBE8760:
.LBE8759:
.LBE8758:
.LBE8806:
	.loc 1 625 0
	movq	%rsi, 112(%rsp)
.LVL1265:
.LBB8807:
	.loc 1 633 0
	divsd	%xmm0, %xmm2
.LVL1266:
.LBB8777:
.LBB8769:
.LBB8761:
	.loc 14 750 0
	movsd	%xmm6, 40(%rsp)
	leaq	272(%rsp), %r12
.LBE8761:
.LBE8769:
.LBE8777:
.LBE8807:
	.loc 1 625 0
	movq	%fs:40, %rax
	movq	%rax, 600(%rsp)
	xorl	%eax, %eax
	leaq	352(%rsp), %r14
.LBB8808:
.LBB8778:
.LBB8770:
.LBB8762:
	.loc 14 750 0
	divsd	%xmm6, %xmm1
	movapd	%xmm2, %xmm0
.LVL1267:
	movsd	%xmm2, (%rsp)
	call	cexp
.LVL1268:
	movsd	%xmm1, 72(%rsp)
.LVL1269:
.LBE8762:
.LBE8770:
.LBE8778:
	.loc 1 633 0
	movsd	%xmm1, 152(%rsp)
.LVL1270:
.LBB8779:
.LBB8780:
.LBB8781:
	.loc 14 750 0
	movsd	.LC23(%rip), %xmm1
	movsd	(%rsp), %xmm2
	divsd	40(%rsp), %xmm1
.LBE8781:
.LBE8780:
.LBE8779:
.LBB8794:
.LBB8771:
.LBB8763:
	movsd	%xmm0, 64(%rsp)
.LBE8763:
.LBE8771:
.LBE8794:
	.loc 1 633 0
	movsd	%xmm0, 144(%rsp)
.LBB8795:
.LBB8788:
.LBB8782:
	.loc 14 750 0
	movapd	%xmm2, %xmm0
	call	cexp
.LVL1271:
.LBE8782:
.LBE8788:
.LBE8795:
	.loc 1 633 0
	movsd	40(%rsp), %xmm6
	movsd	.LC27(%rip), %xmm2
.LBB8796:
.LBB8789:
.LBB8783:
	.loc 14 750 0
	movsd	%xmm1, 88(%rsp)
.LVL1272:
.LBE8783:
.LBE8789:
.LBE8796:
	.loc 1 633 0
	divsd	%xmm6, %xmm2
	.loc 1 634 0
	movsd	%xmm1, 168(%rsp)
.LBB8797:
.LBB8790:
.LBB8784:
	.loc 14 750 0
	movsd	%xmm0, 80(%rsp)
.LBE8784:
.LBE8790:
.LBE8797:
	.loc 1 634 0
	movsd	%xmm0, 160(%rsp)
.LBB8798:
.LBB8772:
.LBB8764:
	.loc 14 750 0
	movsd	.LC28(%rip), %xmm1
.LBE8764:
.LBE8772:
.LBE8798:
	.loc 1 635 0
	movsd	.LC24(%rip), %xmm5
.LBB8799:
.LBB8773:
.LBB8765:
	.loc 14 750 0
	divsd	%xmm6, %xmm1
	movapd	%xmm2, %xmm0
.LBE8765:
.LBE8773:
.LBE8799:
	.loc 1 635 0
	movsd	%xmm5, 208(%rsp)
	.loc 1 636 0
	movsd	%xmm5, 224(%rsp)
.LBB8800:
.LBB8774:
.LBB8766:
	.loc 14 750 0
	movsd	%xmm2, (%rsp)
.LBE8766:
.LBE8774:
.LBE8800:
	.loc 1 635 0
	movsd	.LC25(%rip), %xmm7
	.loc 1 636 0
	movsd	.LC26(%rip), %xmm3
	.loc 1 635 0
	movsd	%xmm7, 216(%rsp)
	.loc 1 636 0
	movsd	%xmm3, 232(%rsp)
.LVL1273:
.LBB8801:
.LBB8775:
.LBB8767:
	.loc 14 750 0
	call	cexp
.LVL1274:
	movsd	%xmm1, 104(%rsp)
.LVL1275:
.LBE8767:
.LBE8775:
.LBE8801:
	.loc 1 633 0
	movsd	%xmm1, 184(%rsp)
.LVL1276:
.LBB8802:
.LBB8791:
.LBB8785:
	.loc 14 750 0
	movsd	.LC29(%rip), %xmm1
	movsd	(%rsp), %xmm2
	divsd	40(%rsp), %xmm1
.LBE8785:
.LBE8791:
.LBE8802:
.LBB8803:
.LBB8776:
.LBB8768:
	movsd	%xmm0, 96(%rsp)
.LBE8768:
.LBE8776:
.LBE8803:
	.loc 1 633 0
	movsd	%xmm0, 176(%rsp)
.LBB8804:
.LBB8792:
.LBB8786:
	.loc 14 750 0
	movapd	%xmm2, %xmm0
	call	cexp
.LVL1277:
.LBE8786:
.LBE8792:
.LBE8804:
.LBE8808:
	.loc 1 638 0
	leaq	336(%rsp), %rdi
	movq	%rbx, %rax
	movl	$32, %ecx
.LBB8809:
	.loc 1 635 0
	movsd	.LC30(%rip), %xmm6
	leaq	208(%rsp), %rbx
.LBE8809:
	.loc 1 638 0
	rep stosq
	.loc 1 639 0
	leaq	272(%rsp), %rdi
	movl	$8, %ecx
.LBB8810:
	.loc 1 635 0
	movsd	%xmm6, 240(%rsp)
	movsd	.LC31(%rip), %xmm6
	.loc 1 636 0
	movsd	.LC32(%rip), %xmm5
	.loc 1 635 0
	movsd	%xmm6, 248(%rsp)
	.loc 1 636 0
	movsd	.LC30(%rip), %xmm6
.LBE8810:
	.loc 1 639 0
	rep stosq
.LBB8811:
.LBB8805:
.LBB8793:
.LBB8787:
	.loc 14 750 0
	movsd	%xmm0, 48(%rsp)
	movsd	%xmm1, 56(%rsp)
.LVL1278:
.LBE8787:
.LBE8793:
.LBE8805:
	.loc 1 634 0
	movsd	%xmm0, 192(%rsp)
	movsd	%xmm1, 200(%rsp)
	.loc 1 636 0
	movsd	%xmm6, 256(%rsp)
	movsd	%xmm5, 264(%rsp)
.LVL1279:
.L618:
.LBE8811:
.LBB8812:
.LBB8646:
.LBB8647:
.LBB8648:
.LBB8649:
.LBB8650:
	.loc 14 1334 0
	pxor	%xmm1, %xmm1
	movsd	.LC15(%rip), %xmm0
	movsd	8(%r13), %xmm3
	movsd	0(%r13), %xmm2
	call	__divdc3
.LVL1280:
.LBE8650:
.LBE8649:
.LBE8648:
.LBE8647:
.LBB8651:
.LBB8652:
	.loc 14 1254 0
	movsd	.LC33(%rip), %xmm6
.LBE8652:
.LBE8651:
	.loc 1 644 0
	testl	%ebp, %ebp
	.loc 1 642 0
	movsd	%xmm0, -16(%r14)
.LBB8655:
.LBB8653:
	.loc 14 1254 0
	movq	$0, 8(%r12)
.LBE8653:
.LBE8655:
	.loc 1 642 0
	movsd	%xmm1, -8(%r14)
.LVL1281:
.LBB8656:
.LBB8654:
	.loc 14 1254 0
	movsd	%xmm6, (%r12)
.LBE8654:
.LBE8656:
	.loc 1 644 0
	je	.L611
.LVL1282:
.LBB8657:
.LBB8658:
.LBB8659:
.LBB8660:
	.loc 14 1323 0 discriminator 1
	movapd	%xmm1, %xmm3
	movapd	%xmm0, %xmm2
	movsd	%xmm1, 16(%rsp)
	movsd	%xmm0, 8(%rsp)
	movsd	72(%rsp), %xmm1
.LVL1283:
	movsd	64(%rsp), %xmm0
.LVL1284:
	call	__muldc3
.LVL1285:
.LBE8660:
.LBE8659:
.LBE8658:
.LBE8657:
.LBB8661:
.LBB8662:
	movapd	%xmm1, %xmm3
	movsd	.LC15(%rip), %xmm2
	xorpd	.LC34(%rip), %xmm3
	subsd	%xmm0, %xmm2
	movsd	8(%rbx), %xmm1
.LVL1286:
	movsd	(%rbx), %xmm0
	call	__muldc3
.LVL1287:
.LBE8662:
.LBE8661:
.LBB8666:
.LBB8667:
.LBB8668:
.LBB8669:
.LBB8670:
	movsd	16(%rsp), %xmm5
	movsd	8(%rsp), %xmm4
.LBE8670:
.LBE8669:
.LBE8668:
.LBE8667:
.LBE8666:
.LBB8747:
.LBB8663:
	movsd	%xmm1, (%rsp)
.LBE8663:
.LBE8747:
.LBB8748:
.LBB8692:
.LBB8685:
.LBB8678:
.LBB8671:
	movapd	%xmm5, %xmm3
.LBE8671:
.LBE8678:
.LBE8685:
.LBE8692:
.LBE8748:
.LBB8749:
.LBB8664:
	movsd	%xmm0, (%rbx)
.LBE8664:
.LBE8749:
.LBB8750:
.LBB8693:
.LBB8686:
.LBB8679:
.LBB8672:
	movapd	%xmm4, %xmm2
	movapd	%xmm4, %xmm0
.LBE8672:
.LBE8679:
.LBE8686:
.LBE8693:
.LBE8750:
.LBB8751:
.LBB8665:
	movsd	%xmm1, 8(%rbx)
.LVL1288:
.LBE8665:
.LBE8751:
.LBB8752:
.LBB8694:
.LBB8687:
.LBB8680:
.LBB8673:
	movapd	%xmm5, %xmm1
	call	__muldc3
.LVL1289:
.LBE8673:
.LBE8680:
.LBE8687:
.LBE8694:
	.loc 1 651 0 discriminator 1
	cmpl	$1, %ebp
	.loc 1 647 0 discriminator 1
	movsd	%xmm0, (%r14)
	movsd	%xmm1, 8(%r14)
	.loc 1 651 0 discriminator 1
	movsd	8(%rsp), %xmm4
	movsd	16(%rsp), %xmm5
	je	.L612
	.loc 1 651 0 is_stmt 0
	movq	%r14, 24(%rsp)
.LVL1290:
.L622:
.LBB8695:
.LBB8696:
.LBB8697:
.LBB8698:
.LBB8699:
.LBB8700:
	.loc 14 1224 0 is_stmt 1
	movsd	-16(%r14), %xmm7
.LVL1291:
.LBE8700:
.LBE8699:
.LBB8701:
.LBB8702:
	.loc 14 1228 0
	movsd	-8(%r14), %xmm6
.LVL1292:
.LBE8702:
.LBE8701:
	.loc 14 1323 0
	movapd	%xmm7, %xmm2
	movsd	80(%rsp), %xmm0
	movapd	%xmm6, %xmm3
	movsd	%xmm6, 16(%rsp)
	movsd	88(%rsp), %xmm1
	movsd	%xmm7, 8(%rsp)
	call	__muldc3
.LVL1293:
.LBE8698:
.LBE8697:
.LBE8696:
.LBE8695:
.LBB8712:
.LBB8713:
	movsd	.LC15(%rip), %xmm4
	xorpd	.LC34(%rip), %xmm1
.LVL1294:
	subsd	%xmm0, %xmm4
	movsd	(%rsp), %xmm3
	movsd	(%rbx), %xmm2
	movapd	%xmm4, %xmm0
	call	__muldc3
.LVL1295:
.LBE8713:
.LBE8712:
.LBB8719:
.LBB8720:
.LBB8721:
.LBB8722:
	movsd	8(%r14), %xmm4
.LBE8722:
.LBE8721:
.LBE8720:
.LBE8719:
.LBB8732:
.LBB8714:
	movsd	%xmm1, (%rsp)
	movsd	%xmm0, (%rbx)
.LBE8714:
.LBE8732:
.LBB8733:
.LBB8729:
.LBB8726:
.LBB8723:
	movapd	%xmm4, %xmm3
.LBE8723:
.LBE8726:
.LBE8729:
.LBE8733:
.LBB8734:
.LBB8715:
	movsd	%xmm1, 8(%rbx)
.LVL1296:
.LBE8715:
.LBE8734:
.LBB8735:
.LBB8730:
.LBB8727:
.LBB8724:
	movsd	8(%rsp), %xmm0
	movsd	16(%rsp), %xmm1
	movsd	(%r14), %xmm2
	movsd	%xmm4, 32(%rsp)
	call	__muldc3
.LVL1297:
.LBE8724:
.LBE8727:
.LBE8730:
.LBE8735:
	.loc 1 651 0
	cmpl	$2, %ebp
	.loc 1 649 0
	movsd	%xmm0, 16(%r14)
	movsd	%xmm1, 24(%r14)
	.loc 1 651 0
	je	.L613
.LVL1298:
.L623:
.LBB8736:
.LBB8709:
.LBB8706:
.LBB8703:
	.loc 14 1323 0
	movsd	96(%rsp), %xmm0
	movsd	16(%rsp), %xmm3
	movsd	8(%rsp), %xmm2
	movsd	104(%rsp), %xmm1
	call	__muldc3
.LVL1299:
.LBE8703:
.LBE8706:
.LBE8709:
.LBE8736:
.LBB8737:
.LBB8716:
	movsd	.LC15(%rip), %xmm6
	xorpd	.LC34(%rip), %xmm1
.LVL1300:
	subsd	%xmm0, %xmm6
	movsd	(%rsp), %xmm3
	movsd	(%rbx), %xmm2
	movapd	%xmm6, %xmm0
	call	__muldc3
.LVL1301:
.LBE8716:
.LBE8737:
.LBB8738:
.LBB8688:
.LBB8681:
.LBB8674:
	movq	24(%rsp), %rax
.LBE8674:
.LBE8681:
.LBE8688:
.LBE8738:
.LBB8739:
.LBB8717:
	movsd	%xmm1, (%rsp)
	movsd	%xmm0, (%rbx)
	movsd	%xmm1, 8(%rbx)
.LVL1302:
.LBE8717:
.LBE8739:
.LBB8740:
.LBB8689:
.LBB8682:
.LBB8675:
	movsd	(%rax), %xmm0
	movsd	8(%rax), %xmm1
	movsd	32(%rsp), %xmm3
	movsd	(%r14), %xmm2
	call	__muldc3
.LVL1303:
.LBE8675:
.LBE8682:
.LBE8689:
.LBE8740:
	.loc 1 651 0
	cmpl	$3, %ebp
	.loc 1 647 0
	movsd	%xmm0, 32(%r14)
	movsd	%xmm1, 40(%r14)
	.loc 1 651 0
	je	.L617
.L614:
.LVL1304:
.LBB8741:
.LBB8710:
.LBB8707:
.LBB8704:
	.loc 14 1323 0
	movsd	48(%rsp), %xmm0
.LBE8704:
.LBE8707:
.LBE8710:
.LBE8741:
.LBE8752:
.LBE8646:
	.loc 1 641 0
	addl	$1, %ebp
.LVL1305:
	addq	$16, %r13
.LBB8755:
.LBB8753:
.LBB8742:
.LBB8711:
.LBB8708:
.LBB8705:
	.loc 14 1323 0
	movsd	16(%rsp), %xmm3
	addq	$16, %r12
	movsd	8(%rsp), %xmm2
	addq	$16, %rbx
	movsd	56(%rsp), %xmm1
	addq	$64, %r14
.LVL1306:
	call	__muldc3
.LVL1307:
.LBE8705:
.LBE8708:
.LBE8711:
.LBE8742:
.LBB8743:
.LBB8718:
	movsd	.LC15(%rip), %xmm7
	xorpd	.LC34(%rip), %xmm1
.LVL1308:
	subsd	%xmm0, %xmm7
	movsd	-16(%rbx), %xmm2
	movsd	(%rsp), %xmm3
	movapd	%xmm7, %xmm0
	call	__muldc3
.LVL1309:
	movsd	%xmm0, -16(%rbx)
	movsd	%xmm1, -8(%rbx)
.LVL1310:
.LBE8718:
.LBE8743:
.LBE8753:
.LBE8755:
	.loc 1 641 0
	cmpl	$4, %ebp
	jne	.L618
.LVL1311:
.L617:
.LBE8812:
	.loc 1 655 0
	leaq	128(%rsp), %rbx
	leaq	336(%rsp), %rcx
	movl	$4, %r8d
	movl	$4, %edx
	movl	$4, %esi
	movl	$101, %edi
	movq	%rbx, %r9
	call	LAPACKE_zgetrf
.LVL1312:
	testl	%eax, %eax
	jne	.L634
	.loc 1 659 0
	subq	$8, %rsp
	.cfi_def_cfa_offset 680
	movl	$4, %r9d
	movl	$1, %ecx
	pushq	$1
	.cfi_def_cfa_offset 688
	movl	$4, %edx
	movl	$78, %esi
	movl	$101, %edi
	leaq	288(%rsp), %rax
.LVL1313:
	pushq	%rax
	.cfi_def_cfa_offset 696
	pushq	%rbx
	.cfi_def_cfa_offset 704
	leaq	368(%rsp), %r8
	call	LAPACKE_zgetrs
.LVL1314:
	addq	$32, %rsp
	.cfi_def_cfa_offset 672
	testl	%eax, %eax
	jne	.L635
.LVL1315:
.LBB8813:
.LBB8814:
.LBB8815:
	.loc 14 1254 0
	pxor	%xmm4, %xmm4
	movsd	.LC15(%rip), %xmm7
.LBE8815:
.LBE8814:
.LBB8827:
.LBB8828:
.LBB8829:
.LBB8830:
	.loc 14 1334 0
	movsd	152(%rsp), %xmm3
	movapd	%xmm7, %xmm0
	movapd	%xmm4, %xmm1
	movsd	144(%rsp), %xmm2
.LBE8830:
.LBE8829:
.LBE8828:
.LBE8827:
.LBB8867:
.LBB8816:
	.loc 14 1254 0
	movsd	%xmm7, 336(%rsp)
	movsd	%xmm4, 344(%rsp)
.LVL1316:
.LBE8816:
.LBE8867:
.LBB8868:
.LBB8855:
.LBB8843:
.LBB8831:
	.loc 14 1334 0
	movsd	%xmm4, 16(%rsp)
	call	__divdc3
.LVL1317:
.LBE8831:
.LBE8843:
.LBE8855:
.LBE8868:
.LBB8869:
.LBB8870:
.LBB8871:
.LBB8872:
	.loc 14 1323 0
	movapd	%xmm1, %xmm3
.LBE8872:
.LBE8871:
.LBE8870:
.LBE8869:
	.loc 1 665 0
	movsd	%xmm0, 352(%rsp)
.LBB8918:
.LBB8903:
.LBB8888:
.LBB8873:
	.loc 14 1323 0
	movapd	%xmm0, %xmm2
.LBE8873:
.LBE8888:
.LBE8903:
.LBE8918:
	.loc 1 665 0
	movsd	%xmm1, 360(%rsp)
.LVL1318:
.LBB8919:
.LBB8904:
.LBB8889:
.LBB8874:
	.loc 14 1323 0
	movsd	%xmm1, 8(%rsp)
	movsd	%xmm0, (%rsp)
	call	__muldc3
.LVL1319:
.LBE8874:
.LBE8889:
.LBE8904:
.LBE8919:
.LBB8920:
.LBB8921:
.LBB8922:
.LBB8923:
	movsd	8(%rsp), %xmm6
	movsd	(%rsp), %xmm5
.LBE8923:
.LBE8922:
.LBE8921:
.LBE8920:
.LBB8957:
.LBB8905:
.LBB8890:
.LBB8875:
	movapd	%xmm0, %xmm2
.LVL1320:
.LBE8875:
.LBE8890:
.LBE8905:
.LBE8957:
.LBB8958:
.LBB8946:
.LBB8935:
.LBB8924:
	movapd	%xmm1, %xmm3
.LBE8924:
.LBE8935:
.LBE8946:
.LBE8958:
	.loc 1 666 0
	movsd	%xmm0, 368(%rsp)
	movsd	%xmm1, 376(%rsp)
.LVL1321:
.LBB8959:
.LBB8947:
.LBB8936:
.LBB8925:
	.loc 14 1323 0
	movapd	%xmm5, %xmm0
.LVL1322:
	movapd	%xmm6, %xmm1
.LVL1323:
	call	__muldc3
.LVL1324:
.LBE8925:
.LBE8936:
.LBE8947:
.LBE8959:
.LBB8960:
.LBB8817:
	.loc 14 1254 0
	movsd	.LC15(%rip), %xmm4
.LBE8817:
.LBE8960:
.LBB8961:
.LBB8856:
.LBB8844:
.LBB8832:
	.loc 14 1334 0
	movsd	168(%rsp), %xmm3
.LBE8832:
.LBE8844:
.LBE8856:
.LBE8961:
.LBB8962:
.LBB8818:
	.loc 14 1254 0
	movsd	%xmm4, 400(%rsp)
	movsd	16(%rsp), %xmm4
.LBE8818:
.LBE8962:
.LBB8963:
.LBB8857:
.LBB8845:
.LBB8833:
	.loc 14 1334 0
	movsd	160(%rsp), %xmm2
.LBE8833:
.LBE8845:
.LBE8857:
.LBE8963:
	.loc 1 667 0
	movsd	%xmm0, 384(%rsp)
	movsd	%xmm1, 392(%rsp)
.LVL1325:
.LBB8964:
.LBB8858:
.LBB8846:
.LBB8834:
	.loc 14 1334 0
	movapd	%xmm4, %xmm1
	movsd	.LC15(%rip), %xmm0
.LBE8834:
.LBE8846:
.LBE8858:
.LBE8964:
.LBB8965:
.LBB8819:
	.loc 14 1254 0
	movsd	%xmm4, 408(%rsp)
.LVL1326:
.LBE8819:
.LBE8965:
.LBB8966:
.LBB8859:
.LBB8847:
.LBB8835:
	.loc 14 1334 0
	call	__divdc3
.LVL1327:
.LBE8835:
.LBE8847:
.LBE8859:
.LBE8966:
.LBB8967:
.LBB8906:
.LBB8891:
.LBB8876:
	.loc 14 1323 0
	movapd	%xmm1, %xmm3
.LBE8876:
.LBE8891:
.LBE8906:
.LBE8967:
	.loc 1 665 0
	movsd	%xmm0, 416(%rsp)
.LBB8968:
.LBB8907:
.LBB8892:
.LBB8877:
	.loc 14 1323 0
	movapd	%xmm0, %xmm2
.LBE8877:
.LBE8892:
.LBE8907:
.LBE8968:
	.loc 1 665 0
	movsd	%xmm1, 424(%rsp)
.LVL1328:
.LBB8969:
.LBB8908:
.LBB8893:
.LBB8878:
	.loc 14 1323 0
	movsd	%xmm1, 8(%rsp)
	movsd	%xmm0, (%rsp)
	call	__muldc3
.LVL1329:
.LBE8878:
.LBE8893:
.LBE8908:
.LBE8969:
.LBB8970:
.LBB8948:
.LBB8937:
.LBB8926:
	movsd	8(%rsp), %xmm6
	movsd	(%rsp), %xmm5
.LBE8926:
.LBE8937:
.LBE8948:
.LBE8970:
.LBB8971:
.LBB8909:
.LBB8894:
.LBB8879:
	movapd	%xmm0, %xmm2
.LVL1330:
.LBE8879:
.LBE8894:
.LBE8909:
.LBE8971:
.LBB8972:
.LBB8949:
.LBB8938:
.LBB8927:
	movapd	%xmm1, %xmm3
.LBE8927:
.LBE8938:
.LBE8949:
.LBE8972:
	.loc 1 666 0
	movsd	%xmm0, 432(%rsp)
	movsd	%xmm1, 440(%rsp)
.LVL1331:
.LBB8973:
.LBB8950:
.LBB8939:
.LBB8928:
	.loc 14 1323 0
	movapd	%xmm5, %xmm0
.LVL1332:
	movapd	%xmm6, %xmm1
.LVL1333:
	call	__muldc3
.LVL1334:
.LBE8928:
.LBE8939:
.LBE8950:
.LBE8973:
.LBB8974:
.LBB8820:
	.loc 14 1254 0
	movsd	16(%rsp), %xmm4
	movsd	.LC15(%rip), %xmm3
.LBE8820:
.LBE8974:
.LBB8975:
.LBB8860:
.LBB8848:
.LBB8836:
	.loc 14 1334 0
	movsd	176(%rsp), %xmm2
.LBE8836:
.LBE8848:
.LBE8860:
.LBE8975:
	.loc 1 667 0
	movsd	%xmm0, 448(%rsp)
	movsd	%xmm1, 456(%rsp)
.LVL1335:
.LBB8976:
.LBB8861:
.LBB8849:
.LBB8837:
	.loc 14 1334 0
	movapd	%xmm4, %xmm1
	movsd	.LC15(%rip), %xmm0
.LBE8837:
.LBE8849:
.LBE8861:
.LBE8976:
.LBB8977:
.LBB8821:
	.loc 14 1254 0
	movsd	%xmm3, 464(%rsp)
.LBE8821:
.LBE8977:
.LBB8978:
.LBB8862:
.LBB8850:
.LBB8838:
	.loc 14 1334 0
	movsd	184(%rsp), %xmm3
.LBE8838:
.LBE8850:
.LBE8862:
.LBE8978:
.LBB8979:
.LBB8822:
	.loc 14 1254 0
	movsd	%xmm4, 472(%rsp)
.LVL1336:
.LBE8822:
.LBE8979:
.LBB8980:
.LBB8863:
.LBB8851:
.LBB8839:
	.loc 14 1334 0
	call	__divdc3
.LVL1337:
.LBE8839:
.LBE8851:
.LBE8863:
.LBE8980:
.LBB8981:
.LBB8910:
.LBB8895:
.LBB8880:
	.loc 14 1323 0
	movapd	%xmm1, %xmm3
.LBE8880:
.LBE8895:
.LBE8910:
.LBE8981:
	.loc 1 665 0
	movsd	%xmm0, 480(%rsp)
.LBB8982:
.LBB8911:
.LBB8896:
.LBB8881:
	.loc 14 1323 0
	movapd	%xmm0, %xmm2
.LBE8881:
.LBE8896:
.LBE8911:
.LBE8982:
	.loc 1 665 0
	movsd	%xmm1, 488(%rsp)
.LVL1338:
.LBB8983:
.LBB8912:
.LBB8897:
.LBB8882:
	.loc 14 1323 0
	movsd	%xmm1, 8(%rsp)
	movsd	%xmm0, (%rsp)
	call	__muldc3
.LVL1339:
.LBE8882:
.LBE8897:
.LBE8912:
.LBE8983:
.LBB8984:
.LBB8951:
.LBB8940:
.LBB8929:
	movsd	(%rsp), %xmm5
	movsd	8(%rsp), %xmm6
.LBE8929:
.LBE8940:
.LBE8951:
.LBE8984:
.LBB8985:
.LBB8913:
.LBB8898:
.LBB8883:
	movapd	%xmm0, %xmm2
.LVL1340:
.LBE8883:
.LBE8898:
.LBE8913:
.LBE8985:
.LBB8986:
.LBB8952:
.LBB8941:
.LBB8930:
	movapd	%xmm1, %xmm3
.LBE8930:
.LBE8941:
.LBE8952:
.LBE8986:
	.loc 1 666 0
	movsd	%xmm0, 496(%rsp)
	movsd	%xmm1, 504(%rsp)
.LVL1341:
.LBB8987:
.LBB8953:
.LBB8942:
.LBB8931:
	.loc 14 1323 0
	movapd	%xmm5, %xmm0
.LVL1342:
	movapd	%xmm6, %xmm1
.LVL1343:
	call	__muldc3
.LVL1344:
.LBE8931:
.LBE8942:
.LBE8953:
.LBE8987:
.LBB8988:
.LBB8823:
	.loc 14 1254 0
	movsd	.LC15(%rip), %xmm6
.LBE8823:
.LBE8988:
	.loc 1 667 0
	movsd	%xmm0, 512(%rsp)
.LBB8989:
.LBB8824:
	.loc 14 1254 0
	movsd	%xmm6, 528(%rsp)
.LBE8824:
.LBE8989:
.LBB8990:
.LBB8864:
.LBB8852:
.LBB8840:
	.loc 14 1334 0
	movapd	%xmm6, %xmm0
.LBE8840:
.LBE8852:
.LBE8864:
.LBE8990:
	.loc 1 667 0
	movsd	%xmm1, 520(%rsp)
.LVL1345:
.LBB8991:
.LBB8825:
	.loc 14 1254 0
	movsd	16(%rsp), %xmm4
.LBE8825:
.LBE8991:
.LBB8992:
.LBB8865:
.LBB8853:
.LBB8841:
	.loc 14 1334 0
	movsd	200(%rsp), %xmm3
	movapd	%xmm4, %xmm1
	movsd	192(%rsp), %xmm2
.LBE8841:
.LBE8853:
.LBE8865:
.LBE8992:
.LBB8993:
.LBB8826:
	.loc 14 1254 0
	movsd	%xmm4, 536(%rsp)
.LVL1346:
.LBE8826:
.LBE8993:
.LBB8994:
.LBB8866:
.LBB8854:
.LBB8842:
	.loc 14 1334 0
	call	__divdc3
.LVL1347:
.LBE8842:
.LBE8854:
.LBE8866:
.LBE8994:
.LBB8995:
.LBB8914:
.LBB8899:
.LBB8884:
	.loc 14 1323 0
	movapd	%xmm1, %xmm3
.LBE8884:
.LBE8899:
.LBE8914:
.LBE8995:
	.loc 1 665 0
	movsd	%xmm0, 544(%rsp)
.LBB8996:
.LBB8915:
.LBB8900:
.LBB8885:
	.loc 14 1323 0
	movapd	%xmm0, %xmm2
.LBE8885:
.LBE8900:
.LBE8915:
.LBE8996:
	.loc 1 665 0
	movsd	%xmm1, 552(%rsp)
.LVL1348:
.LBB8997:
.LBB8916:
.LBB8901:
.LBB8886:
	.loc 14 1323 0
	movsd	%xmm1, 8(%rsp)
	movsd	%xmm0, (%rsp)
	call	__muldc3
.LVL1349:
.LBE8886:
.LBE8901:
.LBE8916:
.LBE8997:
.LBB8998:
.LBB8954:
.LBB8943:
.LBB8932:
	movsd	8(%rsp), %xmm5
	movsd	(%rsp), %xmm4
.LBE8932:
.LBE8943:
.LBE8954:
.LBE8998:
.LBB8999:
.LBB8917:
.LBB8902:
.LBB8887:
	movapd	%xmm0, %xmm2
.LVL1350:
.LBE8887:
.LBE8902:
.LBE8917:
.LBE8999:
.LBB9000:
.LBB8955:
.LBB8944:
.LBB8933:
	movapd	%xmm1, %xmm3
.LBE8933:
.LBE8944:
.LBE8955:
.LBE9000:
	.loc 1 666 0
	movsd	%xmm0, 560(%rsp)
	movsd	%xmm1, 568(%rsp)
.LVL1351:
.LBB9001:
.LBB8956:
.LBB8945:
.LBB8934:
	.loc 14 1323 0
	movapd	%xmm4, %xmm0
.LVL1352:
	movapd	%xmm5, %xmm1
.LVL1353:
	call	__muldc3
.LVL1354:
.LBE8934:
.LBE8945:
.LBE8956:
.LBE9001:
.LBE8813:
	.loc 1 669 0
	leaq	336(%rsp), %rcx
	movq	%rbx, %r9
	movl	$4, %r8d
	movl	$4, %edx
	movl	$4, %esi
	movl	$101, %edi
.LBB9002:
	.loc 1 667 0
	movsd	%xmm0, 576(%rsp)
	movsd	%xmm1, 584(%rsp)
.LVL1355:
.LBE9002:
	.loc 1 669 0
	call	LAPACKE_zgetrf
.LVL1356:
	testl	%eax, %eax
	jne	.L636
	.loc 1 673 0
	subq	$8, %rsp
	.cfi_def_cfa_offset 680
	movl	$4, %r9d
	movl	$1, %ecx
	pushq	$1
	.cfi_def_cfa_offset 688
	movl	$4, %edx
	movl	$78, %esi
	movl	$101, %edi
	leaq	224(%rsp), %rax
.LVL1357:
	pushq	%rax
	.cfi_def_cfa_offset 696
	pushq	%rbx
	.cfi_def_cfa_offset 704
	leaq	368(%rsp), %r8
	call	LAPACKE_zgetrs
.LVL1358:
	addq	$32, %rsp
	.cfi_def_cfa_offset 672
	testl	%eax, %eax
.LBB9003:
.LBB9004:
	.loc 8 98 0
	movsd	40(%rsp), %xmm0
.LBE9004:
.LBE9003:
	.loc 1 673 0
	jne	.L637
	mulsd	.LC38(%rip), %xmm0
.LVL1359:
.LBB9006:
	.loc 1 678 0
	movq	112(%rsp), %rax
.LVL1360:
	.loc 1 679 0
	movsd	208(%rsp), %xmm1
.LBE9006:
	.loc 1 681 0
	movq	120(%rsp), %rdx
.LBB9007:
	.loc 1 679 0
	pxor	%xmm7, %xmm7
	pxor	%xmm3, %xmm3
	.loc 1 678 0
	pxor	%xmm5, %xmm5
	pxor	%xmm4, %xmm4
	.loc 1 679 0
	divsd	%xmm0, %xmm1
	.loc 1 678 0
	cvtsd2ss	272(%rsp), %xmm5
	movss	%xmm5, (%rax)
	cvtsd2ss	288(%rsp), %xmm4
	.loc 1 679 0
	pxor	%xmm5, %xmm5
	.loc 1 678 0
	pxor	%xmm6, %xmm6
	.loc 1 679 0
	cvtsd2ss	%xmm1, %xmm7
	.loc 1 678 0
	cvtsd2ss	304(%rsp), %xmm6
	.loc 1 679 0
	movsd	224(%rsp), %xmm1
	divsd	%xmm0, %xmm1
	movss	%xmm7, (%r15)
.LVL1361:
	.loc 1 678 0
	movss	%xmm4, 4(%rax)
	.loc 1 679 0
	pxor	%xmm4, %xmm4
	.loc 1 678 0
	pxor	%xmm7, %xmm7
	.loc 1 679 0
	cvtsd2ss	%xmm1, %xmm3
	.loc 1 678 0
	cvtsd2ss	320(%rsp), %xmm7
	.loc 1 679 0
	movsd	240(%rsp), %xmm1
	divsd	%xmm0, %xmm1
	movss	%xmm3, 4(%r15)
.LVL1362:
	.loc 1 678 0
	movss	%xmm6, 8(%rax)
	.loc 1 679 0
	cvtsd2ss	%xmm1, %xmm5
	movsd	256(%rsp), %xmm1
	divsd	%xmm0, %xmm1
	movss	%xmm5, 8(%r15)
.LVL1363:
	.loc 1 678 0
	movss	%xmm7, 12(%rax)
	.loc 1 679 0
	movapd	%xmm1, %xmm0
.LBE9007:
	.loc 1 681 0
	movss	(%r15), %xmm1
.LBB9008:
	.loc 1 679 0
	cvtsd2ss	%xmm0, %xmm4
.LBE9008:
	.loc 1 681 0
	movss	4(%r15), %xmm0
.LBB9009:
	.loc 1 679 0
	movss	%xmm4, 12(%r15)
.LVL1364:
.LBE9009:
	.loc 1 681 0
	mulss	(%rax), %xmm1
	subss	%xmm1, %xmm0
	movss	%xmm0, (%rdx)
	.loc 1 682 0
	movss	(%r15), %xmm1
	mulss	4(%rax), %xmm1
	movss	8(%r15), %xmm0
	subss	%xmm1, %xmm0
	movss	%xmm0, 4(%rdx)
	.loc 1 683 0
	movss	(%r15), %xmm1
	mulss	8(%rax), %xmm1
	movss	12(%r15), %xmm0
	subss	%xmm1, %xmm0
	movss	%xmm0, 8(%rdx)
	.loc 1 684 0
	movss	(%r15), %xmm0
	movss	.LC39(%rip), %xmm1
	xorps	%xmm1, %xmm0
	mulss	12(%rax), %xmm0
	.loc 1 685 0
	movq	600(%rsp), %rax
	xorq	%fs:40, %rax
	.loc 1 684 0
	movss	%xmm0, 12(%rdx)
	.loc 1 685 0
	jne	.L638
	addq	$616, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
.LVL1365:
	ret
.LVL1366:
.L611:
	.cfi_restore_state
.LBB9010:
.LBB8756:
.LBB8754:
.LBB8744:
.LBB8690:
.LBB8683:
.LBB8676:
	.loc 14 1323 0
	movsd	344(%rsp), %xmm3
	movsd	336(%rsp), %xmm2
	call	__muldc3
.LVL1367:
	movsd	8(%rbx), %xmm3
.LBE8676:
.LBE8683:
.LBE8690:
.LBE8744:
	.loc 1 647 0
	leaq	352(%rsp), %rax
	movsd	%xmm0, 352(%rsp)
	movsd	%xmm1, 360(%rsp)
	movq	%rax, 24(%rsp)
	movsd	%xmm3, (%rsp)
	jmp	.L622
.LVL1368:
.L613:
.LBB8745:
.LBB8691:
.LBB8684:
.LBB8677:
	.loc 14 1323 0
	movq	24(%rsp), %rax
	movsd	488(%rsp), %xmm3
	movsd	480(%rsp), %xmm2
	movsd	8(%rax), %xmm1
	movsd	(%rax), %xmm0
	call	__muldc3
.LVL1369:
.LBE8677:
.LBE8684:
.LBE8691:
.LBE8745:
	.loc 1 647 0
	movsd	%xmm0, 512(%rsp)
	movsd	%xmm1, 520(%rsp)
	movsd	-16(%r14), %xmm3
	movsd	%xmm3, 8(%rsp)
	movsd	-8(%r14), %xmm3
	movsd	%xmm3, 16(%rsp)
	jmp	.L614
.LVL1370:
.L612:
.LBB8746:
.LBB8731:
.LBB8728:
.LBB8725:
	.loc 14 1323 0
	movsd	424(%rsp), %xmm3
	movapd	%xmm5, %xmm1
	movapd	%xmm4, %xmm0
	movsd	416(%rsp), %xmm2
	call	__muldc3
.LVL1371:
.LBE8725:
.LBE8728:
.LBE8731:
.LBE8746:
	.loc 1 649 0
	movsd	%xmm0, 432(%rsp)
	movq	%r14, 24(%rsp)
	movsd	%xmm1, 440(%rsp)
	movsd	-16(%r14), %xmm3
	movsd	%xmm3, 8(%rsp)
	movsd	-8(%r14), %xmm3
	movsd	%xmm3, 16(%rsp)
	movsd	8(%r14), %xmm3
	movsd	%xmm3, 32(%rsp)
	jmp	.L623
.LVL1372:
.L638:
.LBE8754:
.LBE8756:
.LBE9010:
	.loc 1 685 0
	call	__stack_chk_fail
.LVL1373:
.L637:
.LBB9011:
.LBB9005:
	.loc 8 98 0
	movl	$.LC37, %edx
.LVL1374:
.L631:
.LBE9005:
.LBE9011:
.LBB9012:
.LBB9013:
	movq	stderr(%rip), %rdi
	movl	$1, %esi
	movl	$1, %eax
.LVL1375:
	call	__fprintf_chk
.LVL1376:
.LBE9013:
.LBE9012:
	.loc 1 671 0
	movl	$-1, %edi
	call	exit
.LVL1377:
.L636:
.LBB9015:
.LBB9014:
	.loc 8 98 0
	movsd	40(%rsp), %xmm0
	movl	$.LC36, %edx
	jmp	.L631
.LVL1378:
.L635:
.LBE9014:
.LBE9015:
.LBB9016:
.LBB9017:
	movsd	40(%rsp), %xmm0
	movl	$.LC35, %edx
	jmp	.L631
.LVL1379:
.L634:
.LBE9017:
.LBE9016:
	movsd	40(%rsp), %xmm0
	call	_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_.part.37
.LVL1380:
	.cfi_endproc
.LFE11284:
	.size	_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_, .-_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_
	.section	.text.unlikely._ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_,"axG",@progbits,_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_,comdat
.LCOLDE40:
	.section	.text._ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_,"axG",@progbits,_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_,comdat
.LHOTE40:
	.section	.text.unlikely
	.align 2
.LCOLDB43:
	.text
.LHOTB43:
	.align 2
	.p2align 4,,15
	.globl	_ZN15fastGaussianIIRC2Edd
	.type	_ZN15fastGaussianIIRC2Edd, @function
_ZN15fastGaussianIIRC2Edd:
.LFB11278:
	.loc 1 336 0
	.cfi_startproc
.LVL1381:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
.LBB9032:
.LBB9033:
.LBB9034:
	.loc 1 689 0
	leaq	48(%rdi), %rcx
.LBE9034:
.LBE9033:
	.loc 1 336 0
	movsd	%xmm0, (%rdi)
	movsd	%xmm1, 8(%rdi)
.LVL1382:
.LBB9048:
.LBB9045:
	.loc 1 689 0
	leaq	32(%rdi), %rdx
	leaq	16(%rdi), %rsi
.LBE9045:
.LBE9048:
.LBE9032:
	.loc 1 336 0
	movq	%rdi, %rbx
.LBB9051:
.LBB9049:
.LBB9046:
	.loc 1 689 0
	call	_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_
.LVL1383:
.LBB9035:
	.loc 1 695 0
	movsd	8(%rbx), %xmm0
	movsd	(%rbx), %xmm1
	movsd	.LC41(%rip), %xmm2
	subsd	%xmm0, %xmm1
	andpd	%xmm2, %xmm1
	movsd	.LC42(%rip), %xmm2
	ucomisd	%xmm1, %xmm2
	ja	.L647
.LBB9036:
	.loc 1 700 0
	leaq	96(%rbx), %rcx
	leaq	80(%rbx), %rdx
	leaq	64(%rbx), %rsi
	movq	%rbx, %rdi
.LBE9036:
.LBE9035:
.LBE9046:
.LBE9049:
.LBE9051:
	.loc 1 339 0
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 8
.LVL1384:
.LBB9052:
.LBB9050:
.LBB9047:
.LBB9044:
.LBB9037:
	.loc 1 700 0
	jmp	_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_
.LVL1385:
	.p2align 4,,10
	.p2align 3
.L647:
	.cfi_restore_state
.LBE9037:
.LBB9038:
.LBB9039:
	.loc 5 53 0
	movq	16(%rbx), %rax
	movq	24(%rbx), %rdx
	movq	%rax, 64(%rbx)
	movq	%rdx, 72(%rbx)
.LVL1386:
.LBE9039:
.LBE9038:
.LBB9040:
.LBB9041:
	movq	32(%rbx), %rax
	movq	40(%rbx), %rdx
	movq	%rax, 80(%rbx)
	movq	%rdx, 88(%rbx)
.LVL1387:
.LBE9041:
.LBE9040:
.LBB9042:
.LBB9043:
	movq	48(%rbx), %rax
	movq	56(%rbx), %rdx
	movq	%rax, 96(%rbx)
	movq	%rdx, 104(%rbx)
.LBE9043:
.LBE9042:
.LBE9044:
.LBE9047:
.LBE9050:
.LBE9052:
	.loc 1 339 0
	popq	%rbx
	.cfi_def_cfa_offset 8
.LVL1388:
	ret
	.cfi_endproc
.LFE11278:
	.size	_ZN15fastGaussianIIRC2Edd, .-_ZN15fastGaussianIIRC2Edd
	.section	.text.unlikely
.LCOLDE43:
	.text
.LHOTE43:
	.globl	_ZN15fastGaussianIIRC1Edd
	.set	_ZN15fastGaussianIIRC1Edd,_ZN15fastGaussianIIRC2Edd
	.section	.text.unlikely
	.align 2
.LCOLDB44:
	.text
.LHOTB44:
	.align 2
	.p2align 4,,15
	.globl	_ZN15fastGaussianIIR15getCoefficientsEv
	.type	_ZN15fastGaussianIIR15getCoefficientsEv, @function
_ZN15fastGaussianIIR15getCoefficientsEv:
.LFB11286:
	.loc 1 688 0
	.cfi_startproc
.LVL1389:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	.loc 1 689 0
	leaq	48(%rdi), %rcx
	leaq	32(%rdi), %rdx
	leaq	16(%rdi), %rsi
	.loc 1 688 0
	movq	%rdi, %rbx
	.loc 1 689 0
	movsd	(%rdi), %xmm0
	call	_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_
.LVL1390:
.LBB9064:
	.loc 1 695 0
	movsd	8(%rbx), %xmm0
	movsd	(%rbx), %xmm1
	movsd	.LC41(%rip), %xmm2
	subsd	%xmm0, %xmm1
	andpd	%xmm2, %xmm1
	movsd	.LC42(%rip), %xmm2
	ucomisd	%xmm1, %xmm2
	ja	.L656
.LBB9065:
	.loc 1 700 0
	leaq	96(%rbx), %rcx
	leaq	80(%rbx), %rdx
	leaq	64(%rbx), %rsi
	movq	%rbx, %rdi
.LBE9065:
.LBE9064:
	.loc 1 707 0
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 8
.LVL1391:
.LBB9073:
.LBB9066:
	.loc 1 700 0
	jmp	_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_
.LVL1392:
	.p2align 4,,10
	.p2align 3
.L656:
	.cfi_restore_state
.LBE9066:
.LBB9067:
.LBB9068:
	.loc 5 53 0
	movq	16(%rbx), %rax
	movq	24(%rbx), %rdx
	movq	%rax, 64(%rbx)
	movq	%rdx, 72(%rbx)
.LVL1393:
.LBE9068:
.LBE9067:
.LBB9069:
.LBB9070:
	movq	32(%rbx), %rax
	movq	40(%rbx), %rdx
	movq	%rax, 80(%rbx)
	movq	%rdx, 88(%rbx)
.LVL1394:
.LBE9070:
.LBE9069:
.LBB9071:
.LBB9072:
	movq	48(%rbx), %rax
	movq	56(%rbx), %rdx
	movq	%rax, 96(%rbx)
	movq	%rdx, 104(%rbx)
.LBE9072:
.LBE9071:
.LBE9073:
	.loc 1 707 0
	popq	%rbx
	.cfi_def_cfa_offset 8
.LVL1395:
	ret
	.cfi_endproc
.LFE11286:
	.size	_ZN15fastGaussianIIR15getCoefficientsEv, .-_ZN15fastGaussianIIR15getCoefficientsEv
	.section	.text.unlikely
.LCOLDE44:
	.text
.LHOTE44:
	.section	.rodata.str1.1
.LC45:
	.string	"bilateral_filter.cpp"
	.section	.rodata.str1.8
	.align 8
.LC46:
	.string	"h == dst.rows && w == dst.cols"
	.section	.text.unlikely
.LCOLDB48:
	.text
.LHOTB48:
	.p2align 4,,15
	.globl	_Z9boxFilterRKN2cv3MatERS0_ii
	.type	_Z9boxFilterRKN2cv3MatERS0_ii, @function
_Z9boxFilterRKN2cv3MatERS0_ii:
.LFB11288:
	.loc 1 723 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA11288
.LVL1396:
	.loc 1 724 0
	pxor	%xmm0, %xmm0
	.loc 1 723 0
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	.loc 1 724 0
	cvtsi2sd	%edx, %xmm0
	.loc 1 723 0
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	.loc 1 724 0
	pxor	%xmm1, %xmm1
	.loc 1 723 0
	subq	$280, %rsp
	.cfi_def_cfa_offset 336
	.loc 1 725 0
	movl	8(%rdi), %r14d
	movl	12(%rdi), %ebp
	.loc 1 723 0
	movq	%fs:40, %rax
	movq	%rax, 264(%rsp)
	xorl	%eax, %eax
	.loc 1 724 0
	leal	1(%rcx,%rcx), %eax
	movsd	.LC15(%rip), %xmm5
	.loc 1 726 0
	cmpl	8(%rsi), %r14d
	.loc 1 723 0
	movq	%rdi, 24(%rsp)
	.loc 1 724 0
	cvtsi2sd	%eax, %xmm1
	.loc 1 723 0
	movq	%rsi, 16(%rsp)
	movl	%edx, 56(%rsp)
	.loc 1 724 0
	addsd	%xmm0, %xmm0
	addsd	.LC15(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	divsd	%xmm0, %xmm5
	movsd	%xmm5, 8(%rsp)
.LVL1397:
	.loc 1 726 0
	jne	.L658
	.loc 1 726 0 is_stmt 0 discriminator 2
	cmpl	12(%rsi), %ebp
	jne	.L658
.LBB9178:
.LBB9179:
	.loc 2 402 0 is_stmt 1
	movl	(%rdi), %eax
.LBE9179:
.LBE9178:
.LBB9184:
.LBB9185:
.LBB9186:
.LBB9187:
.LBB9188:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
	movq	%rdi, %r12
.LVL1398:
.LBE9188:
.LBE9187:
.LBB9192:
.LBB9193:
	.loc 2 353 0
	leaq	128(%rsp), %rdx
.LVL1399:
	leaq	160(%rsp), %rdi
.LVL1400:
	movl	$2, %esi
.LVL1401:
	movl	%ecx, %ebx
	movl	$6, %ecx
.LVL1402:
.LBE9193:
.LBE9192:
.LBE9186:
.LBE9185:
.LBE9184:
.LBB9236:
.LBB9180:
	.loc 2 402 0
	movl	%eax, %r13d
	movl	%eax, 32(%rsp)
.LBE9180:
.LBE9236:
.LBB9237:
.LBB9228:
.LBB9213:
.LBB9196:
.LBB9197:
	.loc 2 709 0
	leaq	168(%rsp), %rax
.LBE9197:
.LBE9196:
.LBE9213:
.LBE9228:
.LBE9237:
.LBB9238:
.LBB9181:
	.loc 2 402 0
	andl	$4088, %r13d
.LBE9181:
.LBE9238:
.LBB9239:
.LBB9229:
.LBB9214:
.LBB9199:
.LBB9200:
	.loc 2 62 0
	movq	$0, 208(%rsp)
	movq	$0, 200(%rsp)
.LBE9200:
.LBE9199:
.LBE9214:
.LBE9229:
.LBE9239:
.LBB9240:
.LBB9182:
	.loc 2 402 0
	sarl	$3, %r13d
.LBE9182:
.LBE9240:
.LBB9241:
.LBB9230:
.LBB9215:
.LBB9204:
.LBB9198:
	.loc 2 709 0
	movq	%rax, 224(%rsp)
.LBE9198:
.LBE9204:
.LBB9205:
.LBB9189:
	.loc 2 738 0
	leaq	240(%rsp), %rax
.LBE9189:
.LBE9205:
.LBE9215:
.LBE9230:
.LBE9241:
.LBB9242:
.LBB9183:
	.loc 2 402 0
	leal	1(%r13), %r15d
.LBE9183:
.LBE9242:
.LBB9243:
.LBB9231:
.LBB9216:
.LBB9206:
.LBB9201:
	.loc 2 62 0
	movq	$0, 192(%rsp)
	movq	$0, 176(%rsp)
.LBE9201:
.LBE9206:
.LBB9207:
.LBB9190:
	.loc 2 738 0
	movaps	%xmm0, 240(%rsp)
.LBE9190:
.LBE9207:
.LBE9216:
.LBE9231:
.LBE9243:
	.loc 1 728 0
	imull	%r15d, %ebp
.LVL1403:
.LBB9244:
.LBB9232:
.LBB9217:
.LBB9208:
.LBB9191:
	.loc 2 738 0
	movq	%rax, 232(%rsp)
.LBE9191:
.LBE9208:
.LBB9209:
.LBB9202:
	.loc 2 63 0
	movq	$0, 184(%rsp)
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm0
	.loc 2 64 0
	movq	$0, 216(%rsp)
.LVL1404:
.LBE9202:
.LBE9209:
.LBB9210:
.LBB9194:
	.loc 2 352 0
	movl	%r14d, 128(%rsp)
.LBE9194:
.LBE9210:
.LBB9211:
.LBB9203:
	.loc 2 60 0
	movaps	%xmm0, 160(%rsp)
.LBE9203:
.LBE9211:
.LBB9212:
.LBB9195:
	.loc 2 352 0
	movl	%ebp, 132(%rsp)
.LEHB0:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL1405:
.LEHE0:
.LBE9195:
.LBE9212:
.LBE9217:
.LBB9218:
.LBB9219:
	.loc 2 910 0
	leaq	128(%rsp), %rsi
.LVL1406:
	leaq	160(%rsp), %rdi
.LVL1407:
.LBB9220:
.LBB9221:
.LBB9222:
.LBB9223:
	.loc 13 208 0
	movq	$0, 128(%rsp)
.LVL1408:
	movq	$0, 136(%rsp)
.LVL1409:
	movq	$0, 144(%rsp)
.LVL1410:
	movq	$0, 152(%rsp)
.LVL1411:
.LEHB1:
.LBE9223:
.LBE9222:
.LBE9221:
.LBE9220:
	.loc 2 910 0
	call	_ZN2cv3MataSERKNS_7Scalar_IdEE
.LVL1412:
.LEHE1:
	movq	176(%rsp), %rdi
.LVL1413:
	movl	%r15d, %eax
	movq	16(%r12), %rsi
.LVL1414:
	movq	%rax, %rcx
	movq	%rax, 112(%rsp)
	leaq	(%rdi,%rax,8), %rax
	cmpq	%rax, %rsi
	movq	%rcx, %rax
	setnb	%dl
	addq	%rsi, %rax
	cmpq	%rax, %rdi
	setnb	%al
	orb	%al, %dl
	je	.L746
	cmpl	$20, %r15d
	jbe	.L746
	movq	%rdi, %rax
	salq	$60, %rax
	shrq	$63, %rax
	cmpl	%r15d, %eax
	cmova	%r15d, %eax
	testl	%eax, %eax
	je	.L741
.LBE9219:
.LBE9218:
.LBE9232:
.LBE9244:
.LBB9245:
	.loc 1 733 0
	movzbl	(%rsi), %edx
	pxor	%xmm0, %xmm0
	.loc 1 732 0
	leaq	8(%rdi), %r10
	leaq	1(%rsi), %r9
	movl	$1, %r11d
	.loc 1 733 0
	cvtsi2sd	%edx, %xmm0
	movsd	%xmm0, (%rdi)
.LVL1415:
.L666:
	movl	%r15d, %r12d
.LVL1416:
	movl	%eax, %edx
	.loc 1 732 0
	xorl	%r8d, %r8d
	subl	%eax, %r12d
	leal	-16(%r12), %eax
	shrl	$4, %eax
	addl	$1, %eax
	movl	%eax, 32(%rsp)
	sall	$4, %eax
	movl	%eax, %ecx
	leaq	(%rdi,%rdx,8), %rax
	addq	%rsi, %rdx
.LVL1417:
.L668:
	.loc 1 733 0 discriminator 2
	movdqu	(%rdx), %xmm0
	addl	$1, %r8d
	subq	$-128, %rax
	addq	$16, %rdx
	pmovzxbw	%xmm0, %xmm1
	psrldq	$8, %xmm0
	pmovzxbw	%xmm0, %xmm0
	pmovzxwd	%xmm1, %xmm3
	psrldq	$8, %xmm1
	pmovzxwd	%xmm1, %xmm1
	pmovzxwd	%xmm0, %xmm2
	psrldq	$8, %xmm0
	cvtdq2pd	%xmm3, %xmm4
	pshufd	$238, %xmm3, %xmm3
	pmovzxwd	%xmm0, %xmm0
	movaps	%xmm4, -128(%rax)
	cvtdq2pd	%xmm3, %xmm3
	movaps	%xmm3, -112(%rax)
	cvtdq2pd	%xmm1, %xmm3
	pshufd	$238, %xmm1, %xmm1
	movaps	%xmm3, -96(%rax)
	cvtdq2pd	%xmm1, %xmm1
	movaps	%xmm1, -80(%rax)
	cvtdq2pd	%xmm2, %xmm1
	movaps	%xmm1, -64(%rax)
	pshufd	$238, %xmm2, %xmm1
	cvtdq2pd	%xmm1, %xmm1
	movaps	%xmm1, -48(%rax)
	cvtdq2pd	%xmm0, %xmm1
	pshufd	$238, %xmm0, %xmm0
	movaps	%xmm1, -32(%rax)
	cvtdq2pd	%xmm0, %xmm0
	movaps	%xmm0, -16(%rax)
	cmpl	32(%rsp), %r8d
	jb	.L668
	movl	%ecx, %eax
	addl	%ecx, %r11d
	leaq	(%r10,%rax,8), %rdx
	addq	%r9, %rax
	cmpl	%r12d, %ecx
	je	.L664
.LVL1418:
	.loc 1 733 0 is_stmt 0
	movzbl	(%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 732 0 is_stmt 1
	leal	1(%r11), %ecx
	cmpl	%ecx, %r15d
	.loc 1 733 0
	movsd	%xmm0, (%rdx)
.LVL1419:
	.loc 1 732 0
	jle	.L664
	.loc 1 733 0
	movzbl	1(%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 732 0
	leal	2(%r11), %ecx
	cmpl	%ecx, %r15d
	.loc 1 733 0
	movsd	%xmm0, 8(%rdx)
.LVL1420:
	.loc 1 732 0
	jle	.L664
	.loc 1 733 0
	movzbl	2(%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 732 0
	leal	3(%r11), %ecx
	cmpl	%ecx, %r15d
	.loc 1 733 0
	movsd	%xmm0, 16(%rdx)
.LVL1421:
	.loc 1 732 0
	jle	.L664
	.loc 1 733 0
	movzbl	3(%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 732 0
	leal	4(%r11), %ecx
	cmpl	%ecx, %r15d
	.loc 1 733 0
	movsd	%xmm0, 24(%rdx)
.LVL1422:
	.loc 1 732 0
	jle	.L664
	.loc 1 733 0
	movzbl	4(%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 732 0
	leal	5(%r11), %ecx
	cmpl	%ecx, %r15d
	.loc 1 733 0
	movsd	%xmm0, 32(%rdx)
.LVL1423:
	.loc 1 732 0
	jle	.L664
	.loc 1 733 0
	movzbl	5(%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 732 0
	leal	6(%r11), %ecx
	cmpl	%ecx, %r15d
	.loc 1 733 0
	movsd	%xmm0, 40(%rdx)
.LVL1424:
	.loc 1 732 0
	jle	.L664
	.loc 1 733 0
	movzbl	6(%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 732 0
	leal	7(%r11), %ecx
	cmpl	%ecx, %r15d
	.loc 1 733 0
	movsd	%xmm0, 48(%rdx)
.LVL1425:
	.loc 1 732 0
	jle	.L664
	.loc 1 733 0
	movzbl	7(%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 732 0
	leal	8(%r11), %ecx
	cmpl	%ecx, %r15d
	.loc 1 733 0
	movsd	%xmm0, 56(%rdx)
.LVL1426:
	.loc 1 732 0
	jle	.L664
	.loc 1 733 0
	movzbl	8(%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 732 0
	leal	9(%r11), %ecx
	cmpl	%ecx, %r15d
	.loc 1 733 0
	movsd	%xmm0, 64(%rdx)
.LVL1427:
	.loc 1 732 0
	jle	.L664
	.loc 1 733 0
	movzbl	9(%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 732 0
	leal	10(%r11), %ecx
	cmpl	%ecx, %r15d
	.loc 1 733 0
	movsd	%xmm0, 72(%rdx)
.LVL1428:
	.loc 1 732 0
	jle	.L664
	.loc 1 733 0
	movzbl	10(%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 732 0
	leal	11(%r11), %ecx
	cmpl	%ecx, %r15d
	.loc 1 733 0
	movsd	%xmm0, 80(%rdx)
.LVL1429:
	.loc 1 732 0
	jle	.L664
	.loc 1 733 0
	movzbl	11(%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 732 0
	leal	12(%r11), %ecx
	cmpl	%ecx, %r15d
	.loc 1 733 0
	movsd	%xmm0, 88(%rdx)
.LVL1430:
	.loc 1 732 0
	jle	.L664
	.loc 1 733 0
	movzbl	12(%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 732 0
	leal	13(%r11), %ecx
	cmpl	%ecx, %r15d
	.loc 1 733 0
	movsd	%xmm0, 96(%rdx)
.LVL1431:
	.loc 1 732 0
	jle	.L664
	.loc 1 733 0
	movzbl	13(%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 732 0
	leal	14(%r11), %ecx
	cmpl	%ecx, %r15d
	.loc 1 733 0
	movsd	%xmm0, 104(%rdx)
.LVL1432:
	.loc 1 732 0
	jle	.L664
	.loc 1 733 0
	movzbl	14(%rax), %eax
.LVL1433:
	pxor	%xmm0, %xmm0
	cvtsi2sd	%eax, %xmm0
	movsd	%xmm0, 112(%rdx)
.LVL1434:
.L664:
	movslq	%r13d, %r11
	addq	$1, %r11
.LBE9245:
.LBB9246:
	.loc 1 735 0 discriminator 3
	cmpl	%r15d, %ebp
	leaq	0(,%r11,8), %rcx
	leaq	(%rsi,%r11), %r9
.LVL1435:
	leaq	(%rdi,%rcx), %rdx
.LVL1436:
	jle	.L681
	leaq	-8(%rcx), %r8
	movslq	%r15d, %rax
	leaq	0(,%rax,8), %r12
	movl	%ebp, %eax
	movq	%r8, 96(%rsp)
	leaq	128(%rcx), %r8
	subl	%r13d, %eax
	movq	%r12, %r10
	subl	$1, %eax
	movq	%r8, 40(%rsp)
	subq	%r12, %r8
	negq	%r10
	cmpq	%r8, %rcx
	setge	48(%rsp)
	subq	%r12, %rcx
	movq	%rcx, 32(%rsp)
	movl	%eax, %ecx
	addq	%r11, %rcx
	leaq	(%rsi,%rcx), %r8
	leaq	(%rdi,%rcx,8), %rcx
	cmpq	%r8, %rdx
	setnb	%r8b
	cmpq	%rcx, %r9
	setnb	%cl
	orb	%cl, %r8b
	je	.L675
	movq	40(%rsp), %r8
	cmpq	%r8, 32(%rsp)
	setge	%cl
	orb	48(%rsp), %cl
	cmpl	$19, %eax
	seta	%r8b
	testb	%r8b, %cl
	je	.L675
	movq	32(%rsp), %rcx
	addq	%rdi, %rcx
	salq	$60, %rcx
	shrq	$63, %rcx
	cmpl	%eax, %ecx
	cmova	%eax, %ecx
	testl	%ecx, %ecx
	movl	%ecx, %r8d
	je	.L742
	.loc 1 736 0
	movzbl	(%r9), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	.loc 1 735 0
	leal	2(%r13), %ecx
	movl	%ecx, 32(%rsp)
.LVL1437:
	leaq	8(%rdx), %rcx
.LVL1438:
	movq	%rcx, 48(%rsp)
.LVL1439:
	leaq	1(%r9), %rcx
.LVL1440:
	movq	%rcx, 40(%rsp)
.LVL1441:
	.loc 1 736 0
	addsd	(%rdx,%r10), %xmm0
	movsd	%xmm0, (%rdx)
.LVL1442:
.L676:
	subl	%r8d, %eax
	movl	%r8d, %ecx
	movl	%eax, 80(%rsp)
	subl	$16, %eax
	leaq	(%rcx,%r11), %r8
	shrl	$4, %eax
	movq	%rcx, 88(%rsp)
	movq	88(%rsp), %r11
	addl	$1, %eax
	leaq	(%rdi,%r8,8), %rcx
	addq	%rsi, %r8
	movl	%eax, 72(%rsp)
	sall	$4, %eax
	movl	%eax, 64(%rsp)
	movq	96(%rsp), %rax
	subq	%r12, %rax
	leaq	8(%rax,%r11,8), %rax
	.loc 1 735 0
	xorl	%r11d, %r11d
	addq	%rdi, %rax
.LVL1443:
.L678:
	.loc 1 736 0 discriminator 2
	movdqu	(%r8), %xmm1
	addl	$1, %r11d
	subq	$-128, %rcx
	addq	$16, %r8
	subq	$-128, %rax
	pmovzxbw	%xmm1, %xmm0
	psrldq	$8, %xmm1
	pmovzxbw	%xmm1, %xmm2
	pmovzxwd	%xmm0, %xmm4
	psrldq	$8, %xmm0
	pmovzxwd	%xmm0, %xmm1
	movdqa	%xmm2, %xmm0
	pmovzxwd	%xmm2, %xmm3
	psrldq	$8, %xmm0
	pmovzxwd	%xmm0, %xmm0
	cvtdq2pd	%xmm1, %xmm6
	pshufd	$238, %xmm3, %xmm2
	addpd	-96(%rax), %xmm6
	cvtdq2pd	%xmm3, %xmm8
	cvtdq2pd	%xmm0, %xmm7
	pshufd	$238, %xmm4, %xmm5
	cvtdq2pd	%xmm4, %xmm3
	pshufd	$238, %xmm1, %xmm1
	addpd	-64(%rax), %xmm8
	pshufd	$238, %xmm0, %xmm0
	cvtdq2pd	%xmm2, %xmm2
	addpd	-32(%rax), %xmm7
	cvtdq2pd	%xmm5, %xmm5
	cvtdq2pd	%xmm1, %xmm1
	addpd	-112(%rax), %xmm5
	cvtdq2pd	%xmm0, %xmm0
	addpd	-80(%rax), %xmm1
	addpd	-48(%rax), %xmm2
	addpd	-16(%rax), %xmm0
	addpd	-128(%rax), %xmm3
	movups	%xmm5, -112(%rcx)
	movups	%xmm6, -96(%rcx)
	movups	%xmm1, -80(%rcx)
	movups	%xmm3, -128(%rcx)
	movups	%xmm8, -64(%rcx)
	movups	%xmm2, -48(%rcx)
	movups	%xmm7, -32(%rcx)
	movups	%xmm0, -16(%rcx)
	cmpl	%r11d, 72(%rsp)
	ja	.L678
	movl	64(%rsp), %r11d
	movq	48(%rsp), %rax
	movl	32(%rsp), %ecx
	movl	%r11d, %r8d
	leaq	(%rax,%r8,8), %rax
	addl	%r11d, %ecx
	addq	40(%rsp), %r8
	cmpl	80(%rsp), %r11d
	je	.L682
.LVL1444:
	.loc 1 736 0 is_stmt 0
	movzbl	(%r8), %r11d
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r11d, %xmm0
	.loc 1 735 0 is_stmt 1
	leal	1(%rcx), %r11d
	cmpl	%r11d, %ebp
	.loc 1 736 0
	addsd	(%rax,%r10), %xmm0
	movsd	%xmm0, (%rax)
.LVL1445:
	.loc 1 735 0
	jle	.L682
	.loc 1 736 0
	movq	%rax, %r11
	pxor	%xmm0, %xmm0
	subq	%r12, %r11
	movq	%r11, %r12
	movzbl	1(%r8), %r11d
	cvtsi2sd	%r11d, %xmm0
	.loc 1 735 0
	leal	2(%rcx), %r11d
	cmpl	%r11d, %ebp
	.loc 1 736 0
	addsd	8(%r12), %xmm0
	movsd	%xmm0, 8(%rax)
.LVL1446:
	.loc 1 735 0
	jle	.L682
	.loc 1 736 0
	movzbl	2(%r8), %r11d
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r11d, %xmm0
	.loc 1 735 0
	leal	3(%rcx), %r11d
	cmpl	%r11d, %ebp
	.loc 1 736 0
	addsd	16(%rax,%r10), %xmm0
	movsd	%xmm0, 16(%rax)
.LVL1447:
	.loc 1 735 0
	jle	.L682
	.loc 1 736 0
	movzbl	3(%r8), %r11d
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r11d, %xmm0
	.loc 1 735 0
	leal	4(%rcx), %r11d
	cmpl	%r11d, %ebp
	.loc 1 736 0
	addsd	24(%rax,%r10), %xmm0
	movsd	%xmm0, 24(%rax)
.LVL1448:
	.loc 1 735 0
	jle	.L682
	.loc 1 736 0
	movzbl	4(%r8), %r11d
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r11d, %xmm0
	.loc 1 735 0
	leal	5(%rcx), %r11d
	cmpl	%r11d, %ebp
	.loc 1 736 0
	addsd	32(%rax,%r10), %xmm0
	movsd	%xmm0, 32(%rax)
.LVL1449:
	.loc 1 735 0
	jle	.L682
	.loc 1 736 0
	movzbl	5(%r8), %r11d
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r11d, %xmm0
	.loc 1 735 0
	leal	6(%rcx), %r11d
	cmpl	%r11d, %ebp
	.loc 1 736 0
	addsd	40(%rax,%r10), %xmm0
	movsd	%xmm0, 40(%rax)
.LVL1450:
	.loc 1 735 0
	jle	.L682
	.loc 1 736 0
	movzbl	6(%r8), %r11d
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r11d, %xmm0
	.loc 1 735 0
	leal	7(%rcx), %r11d
	cmpl	%r11d, %ebp
	.loc 1 736 0
	addsd	48(%rax,%r10), %xmm0
	movsd	%xmm0, 48(%rax)
.LVL1451:
	.loc 1 735 0
	jle	.L682
	.loc 1 736 0
	movzbl	7(%r8), %r11d
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r11d, %xmm0
	.loc 1 735 0
	leal	8(%rcx), %r11d
	cmpl	%r11d, %ebp
	.loc 1 736 0
	addsd	56(%rax,%r10), %xmm0
	movsd	%xmm0, 56(%rax)
.LVL1452:
	.loc 1 735 0
	jle	.L682
	.loc 1 736 0
	movzbl	8(%r8), %r11d
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r11d, %xmm0
	.loc 1 735 0
	leal	9(%rcx), %r11d
	cmpl	%r11d, %ebp
	.loc 1 736 0
	addsd	64(%rax,%r10), %xmm0
	movsd	%xmm0, 64(%rax)
.LVL1453:
	.loc 1 735 0
	jle	.L682
	.loc 1 736 0
	movzbl	9(%r8), %r11d
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r11d, %xmm0
	.loc 1 735 0
	leal	10(%rcx), %r11d
	cmpl	%r11d, %ebp
	.loc 1 736 0
	addsd	72(%rax,%r10), %xmm0
	movsd	%xmm0, 72(%rax)
.LVL1454:
	.loc 1 735 0
	jle	.L682
	.loc 1 736 0
	movzbl	10(%r8), %r11d
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r11d, %xmm0
	.loc 1 735 0
	leal	11(%rcx), %r11d
	cmpl	%r11d, %ebp
	.loc 1 736 0
	addsd	80(%rax,%r10), %xmm0
	movsd	%xmm0, 80(%rax)
.LVL1455:
	.loc 1 735 0
	jle	.L682
	.loc 1 736 0
	movzbl	11(%r8), %r11d
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r11d, %xmm0
	.loc 1 735 0
	leal	12(%rcx), %r11d
	cmpl	%r11d, %ebp
	.loc 1 736 0
	addsd	88(%rax,%r10), %xmm0
	movsd	%xmm0, 88(%rax)
.LVL1456:
	.loc 1 735 0
	jle	.L682
	.loc 1 736 0
	movzbl	12(%r8), %r11d
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r11d, %xmm0
	.loc 1 735 0
	leal	13(%rcx), %r11d
	cmpl	%r11d, %ebp
	.loc 1 736 0
	addsd	96(%rax,%r10), %xmm0
	movsd	%xmm0, 96(%rax)
.LVL1457:
	.loc 1 735 0
	jle	.L682
	.loc 1 736 0
	movzbl	13(%r8), %r11d
	pxor	%xmm0, %xmm0
	.loc 1 735 0
	addl	$14, %ecx
.LVL1458:
	cmpl	%ecx, %ebp
	.loc 1 736 0
	cvtsi2sd	%r11d, %xmm0
	addsd	104(%rax,%r10), %xmm0
	movsd	%xmm0, 104(%rax)
.LVL1459:
	.loc 1 735 0
	jle	.L682
	.loc 1 736 0
	movzbl	14(%r8), %ecx
.LVL1460:
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	addsd	112(%rax,%r10), %xmm0
	movsd	%xmm0, 112(%rax)
.LVL1461:
.L682:
	movl	%ebp, %eax
	subl	%r13d, %eax
	subl	$2, %eax
	addq	$1, %rax
	leaq	(%rdx,%rax,8), %rdx
	addq	%rax, %r9
.L681:
.LVL1462:
.LBE9246:
.LBB9247:
	.loc 1 738 0 discriminator 1
	cmpl	$1, %r14d
	jle	.L688
	movslq	%ebp, %rax
.LBB9248:
	.loc 1 741 0 discriminator 1
	movl	$1, %r12d
	salq	$3, %rax
	negq	%rax
	movq	%rax, %r13
	movslq	%r15d, %rax
	salq	$3, %rax
	movq	%rax, %rcx
	negq	%rcx
	movq	%rcx, 32(%rsp)
	movq	%r13, %rcx
	subq	%rax, %rcx
	leal	-1(%rbp), %eax
	movq	%rcx, 40(%rsp)
	addq	$1, %rax
	movq	%rax, 64(%rsp)
	salq	$3, %rax
	movq	%rax, 48(%rsp)
.LVL1463:
.L689:
	.loc 1 739 0 discriminator 1
	testl	%ebp, %ebp
	jle	.L684
	.loc 1 741 0
	movq	32(%rsp), %r11
	movq	40(%rsp), %r10
	.loc 1 739 0
	xorl	%eax, %eax
	leaq	(%rdx,%r13), %rcx
	.loc 1 741 0
	addq	%rdx, %r11
	addq	%rdx, %r10
	jmp	.L687
.LVL1464:
.L770:
	movsd	(%r11,%rax,8), %xmm0
	movzbl	(%r9,%rax), %r8d
	addsd	(%rcx,%rax,8), %xmm0
	movapd	%xmm0, %xmm1
	pxor	%xmm0, %xmm0
	subsd	(%r10,%rax,8), %xmm1
	cvtsi2sd	%r8d, %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, (%rdx,%rax,8)
.LVL1465:
	addq	$1, %rax
.LVL1466:
	.loc 1 739 0
	cmpl	%eax, %ebp
	jle	.L769
.LVL1467:
.L687:
	.loc 1 740 0
	cmpl	%eax, %r15d
	jl	.L770
	.loc 1 743 0
	movzbl	(%r9,%rax), %r8d
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r8d, %xmm0
	addsd	(%rcx,%rax,8), %xmm0
	movsd	%xmm0, (%rdx,%rax,8)
.LVL1468:
	addq	$1, %rax
.LVL1469:
	.loc 1 739 0
	cmpl	%eax, %ebp
	jg	.L687
.LVL1470:
.L769:
	addq	48(%rsp), %rdx
	addq	64(%rsp), %r9
.L684:
.LBE9248:
	.loc 1 738 0 discriminator 2
	addl	$1, %r12d
.LVL1471:
	cmpl	%r12d, %r14d
	jne	.L689
.LVL1472:
.L688:
.LBE9247:
	.loc 1 750 0
	movl	56(%rsp), %ecx
.LBB9249:
	.loc 1 751 0
	subl	%ebx, %r14d
.LVL1473:
	movl	%r14d, 32(%rsp)
.LBE9249:
	.loc 1 750 0
	imull	%r15d, %ecx
.LVL1474:
.LBB9284:
	.loc 1 751 0
	cmpl	%r14d, %ebx
	jge	.L771
	movslq	%ecx, %rax
.LBB9250:
.LBB9251:
.LBB9252:
	.loc 1 755 0
	pxor	%xmm5, %xmm5
	leaq	0(,%rax,8), %rsi
	movq	%rax, 72(%rsp)
	movl	%ebp, %r11d
	leal	(%rcx,%r15), %r12d
	subl	%ecx, %r11d
	.loc 1 757 0
	movl	%ebx, 40(%rsp)
	movq	%rsi, %r10
	movq	%rsi, 104(%rsp)
	movq	16(%rsp), %rsi
	movapd	%xmm5, %xmm2
	movl	%ebp, 124(%rsp)
	movq	16(%rsi), %rdx
.LVL1475:
	movl	%ebp, %esi
	imull	%ebx, %esi
	movslq	%esi, %rsi
	leaq	(%rax,%rsi), %r8
	.loc 1 760 0
	subq	%rax, %rsi
	leaq	0(,%r8,8), %r14
	movq	%r8, 64(%rsp)
	.loc 1 757 0
	movq	%rdi, %r8
	movq	%r14, 80(%rsp)
	.loc 1 760 0
	movq	%rax, %r14
	movslq	%r15d, %rax
	subq	%rax, %rsi
	leaq	0(,%rsi,8), %rax
	movq	%rax, 56(%rsp)
	leal	1(%rbx), %eax
	movl	%eax, 88(%rsp)
	.loc 1 757 0
	imull	%ebp, %eax
	movslq	%eax, %rsi
	movq	%r14, %rax
	subq	%rsi, %rax
	addq	112(%rsp), %rsi
	salq	$3, %rax
	movq	%rax, 96(%rsp)
	movslq	%ebx, %rax
	movq	%rax, 48(%rsp)
	movslq	%ebp, %rax
	negq	%rsi
	salq	$3, %rsi
	movq	%rsi, %r14
	subq	%r10, %r14
	movq	%r14, 112(%rsp)
.LVL1476:
.L702:
.LBE9252:
.LBE9251:
.LBB9278:
.LBB9279:
	.loc 2 430 0
	movq	232(%rsp), %rsi
.LBE9279:
.LBE9278:
	.loc 1 752 0
	movq	48(%rsp), %rdi
	imulq	(%rsi), %rdi
	movq	%rdi, %rsi
	addq	104(%rsp), %rsi
.LBB9280:
	.loc 1 754 0
	cmpl	%r11d, %ecx
.LBE9280:
	.loc 1 752 0
	leaq	(%r8,%rsi), %rdi
.LVL1477:
	.loc 1 753 0
	movq	64(%rsp), %rsi
	leaq	(%rdx,%rsi), %r9
.LVL1478:
.LBB9281:
	.loc 1 754 0
	jge	.L690
	cmpl	40(%rsp), %ebx
	jl	.L691
	movq	80(%rsp), %r10
	movq	56(%rsp), %r8
	movl	%ecx, %esi
	addq	%r10, %rdi
.LVL1479:
	.p2align 4,,10
	.p2align 3
.L696:
.LBB9273:
	.loc 1 759 0
	cmpl	%r12d, %esi
	.loc 1 755 0
	movapd	%xmm2, %xmm1
	.loc 1 759 0
	jl	.L692
	.loc 1 760 0
	movq	%rdi, %rdx
	subq	%r10, %rdx
.LVL1480:
	movsd	(%rdx,%r8), %xmm1
.LVL1481:
.L692:
.LBB9253:
.LBB9254:
.LBB9255:
.LBB9256:
.LBB9257:
.LBB9258:
	.loc 3 63 0
	movsd	(%rdi), %xmm0
	subsd	%xmm1, %xmm0
	addsd	%xmm2, %xmm0
	mulsd	8(%rsp), %xmm0
.LBE9258:
.LBE9257:
.LBB9260:
.LBB9261:
	.loc 3 827 0
	cvtsd2si	%xmm0, %edx
.LVL1482:
.LBE9261:
.LBE9260:
.LBE9256:
.LBE9255:
.LBB9266:
.LBB9267:
	.loc 13 132 0
	cmpl	$255, %edx
	jbe	.L694
	testl	%edx, %edx
	setg	%dl
.LVL1483:
	negl	%edx
.L694:
.LVL1484:
.LBE9267:
.LBE9266:
.LBE9254:
.LBE9253:
.LBE9273:
	.loc 1 754 0
	addl	$1, %esi
.LVL1485:
.LBB9274:
	.loc 1 765 0
	movb	%dl, (%r9)
	addq	$8, %rdi
.LVL1486:
.LBE9274:
	.loc 1 754 0
	addq	$1, %r9
.LVL1487:
	cmpl	%r11d, %esi
	jne	.L696
.LVL1488:
.L695:
	movq	16(%rsp), %rdi
	movq	16(%rdi), %rdx
.L690:
.LBE9281:
.LBE9250:
	.loc 1 751 0 discriminator 2
	addl	$1, 40(%rsp)
.LVL1489:
	addq	$1, 48(%rsp)
	movl	40(%rsp), %edi
.LVL1490:
	addq	%rax, 64(%rsp)
	cmpl	32(%rsp), %edi
	je	.L701
	movq	176(%rsp), %r8
	jmp	.L702
.LVL1491:
.L658:
.LBE9284:
	.loc 1 726 0 discriminator 3
	movl	$_ZZ9boxFilterRKN2cv3MatERS0_iiE19__PRETTY_FUNCTION__, %ecx
.LVL1492:
	movl	$726, %edx
.LVL1493:
	movl	$.LC45, %esi
.LVL1494:
	movl	$.LC46, %edi
.LVL1495:
	call	__assert_fail
.LVL1496:
	.p2align 4,,10
	.p2align 3
.L691:
	movq	112(%rsp), %rsi
.LBB9285:
.LBB9283:
.LBB9282:
	.loc 1 754 0
	movl	%ecx, %r8d
	xorl	%edx, %edx
	leaq	(%rdi,%rsi), %r14
	movq	80(%rsp), %rsi
	leaq	(%rdi,%rsi), %r13
	movq	96(%rsp), %rsi
	leaq	(%rdi,%rsi), %rbp
.LBB9275:
	.loc 1 760 0
	addq	56(%rsp), %rdi
.LVL1497:
	jmp	.L700
.LVL1498:
	.p2align 4,,10
	.p2align 3
.L697:
	movsd	(%rdi,%rdx,8), %xmm4
.LVL1499:
	.loc 1 763 0
	movsd	(%r14,%rdx,8), %xmm1
.LVL1500:
.L739:
.LBB9272:
.LBB9271:
.LBB9269:
.LBB9265:
.LBB9263:
.LBB9259:
	.loc 3 63 0
	movsd	0(%r13,%rdx,8), %xmm0
	subsd	%xmm3, %xmm0
	subsd	%xmm4, %xmm0
	addsd	%xmm1, %xmm0
	mulsd	8(%rsp), %xmm0
.LBE9259:
.LBE9263:
.LBB9264:
.LBB9262:
	.loc 3 827 0
	cvtsd2si	%xmm0, %esi
.LVL1501:
.LBE9262:
.LBE9264:
.LBE9265:
.LBE9269:
.LBB9270:
.LBB9268:
	.loc 13 132 0
	cmpl	$255, %esi
	movl	%esi, %r10d
	jbe	.L699
	testl	%esi, %esi
	setg	%r10b
	negl	%r10d
.L699:
.LVL1502:
.LBE9268:
.LBE9270:
.LBE9271:
.LBE9272:
.LBE9275:
	.loc 1 754 0
	addl	$1, %r8d
.LVL1503:
.LBB9276:
	.loc 1 765 0
	movb	%r10b, (%r9,%rdx)
	addq	$1, %rdx
.LVL1504:
.LBE9276:
	.loc 1 754 0
	cmpl	%r11d, %r8d
	je	.L695
.LVL1505:
.L700:
.LBB9277:
	.loc 1 759 0
	cmpl	%r12d, %r8d
	.loc 1 757 0
	movsd	0(%rbp,%rdx,8), %xmm3
.LVL1506:
	.loc 1 759 0
	jge	.L697
	.loc 1 755 0
	movapd	%xmm5, %xmm4
	movapd	%xmm5, %xmm1
	jmp	.L739
.LVL1507:
.L701:
.LBE9277:
.LBE9282:
.LBE9283:
.LBE9285:
.LBB9286:
	.loc 1 768 0
	testl	%ebx, %ebx
	movl	124(%rsp), %ebp
.LVL1508:
	jle	.L737
	movq	24(%rsp), %rdi
.LVL1509:
	movq	16(%rdi), %rsi
	movl	32(%rsp), %edi
.LVL1510:
.L738:
	imull	%ebp, %edi
	xorl	%r14d, %r14d
	movl	$0, 8(%rsp)
.LVL1511:
	movslq	%edi, %rdi
	movq	%rdi, 40(%rsp)
.LVL1512:
.L716:
.LBB9287:
	.loc 1 769 0
	leaq	(%rsi,%r14), %r10
.LVL1513:
	.loc 1 770 0
	leaq	(%rdx,%r14), %r12
.LVL1514:
.LBB9288:
	.loc 1 773 0
	testl	%ebp, %ebp
.LBE9288:
	.loc 1 771 0
	leaq	(%r10,%rdi), %r11
.LVL1515:
	.loc 1 772 0
	leaq	(%r12,%rdi), %r13
.LVL1516:
.LBB9315:
	.loc 1 773 0
	jle	.L714
	pxor	%xmm3, %xmm3
	xorl	%r9d, %r9d
	cvtsi2sd	%ebx, %xmm3
	addsd	.LC15(%rip), %xmm3
.LVL1517:
	.p2align 4,,10
	.p2align 3
.L715:
.LBB9289:
	.loc 1 776 0
	movzbl	(%r10,%r9), %edx
	pxor	%xmm2, %xmm2
	.loc 1 777 0
	pxor	%xmm1, %xmm1
	leaq	(%r9,%rax), %rsi
	movl	$1, %edi
	addq	%r10, %rsi
	.loc 1 776 0
	cvtsi2sd	%edx, %xmm2
.LVL1518:
	.loc 1 777 0
	movzbl	(%r11,%r9), %edx
	cvtsi2sd	%edx, %xmm1
.LVL1519:
	movq	%r9, %rdx
	subq	%rax, %rdx
	addq	%r11, %rdx
.LVL1520:
	.p2align 4,,10
	.p2align 3
.L709:
.LBB9290:
	.loc 1 779 0 discriminator 2
	movzbl	(%rsi), %r8d
	pxor	%xmm0, %xmm0
	.loc 1 778 0 discriminator 2
	addl	$1, %edi
.LVL1521:
	addq	%rax, %rsi
	.loc 1 779 0 discriminator 2
	cvtsi2sd	%r8d, %xmm0
	.loc 1 780 0 discriminator 2
	movzbl	(%rdx), %r8d
	subq	%rax, %rdx
	.loc 1 778 0 discriminator 2
	cmpl	%edi, %ebx
	.loc 1 779 0 discriminator 2
	addsd	%xmm0, %xmm2
.LVL1522:
	.loc 1 780 0 discriminator 2
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r8d, %xmm0
	addsd	%xmm0, %xmm1
.LVL1523:
	.loc 1 778 0 discriminator 2
	jge	.L709
.LVL1524:
.LBE9290:
.LBB9291:
.LBB9292:
.LBB9293:
.LBB9294:
.LBB9295:
.LBB9296:
	.loc 3 63 0
	divsd	%xmm3, %xmm2
.LVL1525:
.LBE9296:
.LBE9295:
.LBB9297:
.LBB9298:
	.loc 3 827 0
	cvtsd2si	%xmm2, %edx
.LVL1526:
.LBE9298:
.LBE9297:
.LBE9294:
.LBE9293:
.LBB9299:
.LBB9300:
	.loc 13 132 0
	cmpl	$255, %edx
	movl	%edx, %esi
	jbe	.L711
	testl	%edx, %edx
	setg	%sil
	negl	%esi
.L711:
.LVL1527:
.LBE9300:
.LBE9299:
.LBE9292:
.LBE9291:
.LBB9301:
.LBB9302:
.LBB9303:
.LBB9304:
.LBB9305:
.LBB9306:
	.loc 3 63 0
	divsd	%xmm3, %xmm1
.LVL1528:
.LBE9306:
.LBE9305:
.LBE9304:
.LBE9303:
.LBE9302:
.LBE9301:
	.loc 1 782 0
	movb	%sil, (%r12,%r9)
.LBB9314:
.LBB9313:
.LBB9310:
.LBB9309:
.LBB9307:
.LBB9308:
	.loc 3 827 0
	cvtsd2si	%xmm1, %edx
.LVL1529:
.LBE9308:
.LBE9307:
.LBE9309:
.LBE9310:
.LBB9311:
.LBB9312:
	.loc 13 132 0
	cmpl	$255, %edx
	movl	%edx, %esi
	jbe	.L713
	testl	%edx, %edx
	setg	%sil
	negl	%esi
.L713:
.LVL1530:
.LBE9312:
.LBE9311:
.LBE9313:
.LBE9314:
	.loc 1 783 0
	movb	%sil, 0(%r13,%r9)
.LVL1531:
	addq	$1, %r9
.LVL1532:
.LBE9289:
	.loc 1 773 0
	cmpl	%r9d, %ebp
	jg	.L715
.LVL1533:
.L714:
.LBE9315:
.LBE9287:
	.loc 1 768 0
	addl	$1, 8(%rsp)
.LVL1534:
	addq	%rax, %r14
	movl	8(%rsp), %edi
.LVL1535:
	cmpl	%edi, %ebx
	jle	.L772
	movq	24(%rsp), %rdi
.LVL1536:
	movq	16(%rdi), %rsi
	movq	16(%rsp), %rdi
	movq	16(%rdi), %rdx
	movq	40(%rsp), %rdi
	jmp	.L716
.LVL1537:
.L772:
.LBE9286:
.LBB9316:
	.loc 1 786 0 discriminator 1
	cmpl	32(%rsp), %ebx
	jge	.L718
	movslq	%ecx, %rdi
.LVL1538:
	movq	%rdi, 72(%rsp)
	movq	16(%rsp), %rdi
	movq	16(%rdi), %rdx
	leal	1(%rbx), %edi
	movl	%edi, 88(%rsp)
.LVL1539:
.L737:
	movq	%rax, %rdi
	subq	72(%rsp), %rdi
	movslq	%r15d, %r9
	imull	%ebp, %ebx
.LVL1540:
	movl	88(%rsp), %r15d
	leal	1(%rcx), %r10d
	movq	%rdi, 8(%rsp)
	movslq	%ebx, %r14
.LVL1541:
.L727:
.LBB9317:
	.loc 1 787 0
	movq	24(%rsp), %rdi
	movq	%r14, %rbx
	.loc 1 788 0
	leaq	(%rdx,%r14), %r12
	.loc 1 787 0
	addq	16(%rdi), %rbx
.LVL1542:
	.loc 1 789 0
	movq	8(%rsp), %rdi
.LBB9318:
	.loc 1 791 0
	testl	%ecx, %ecx
.LBE9318:
	.loc 1 790 0
	leaq	(%r12,%rdi), %r13
	.loc 1 789 0
	leaq	(%rbx,%rdi), %rbp
.LVL1543:
.LBB9345:
	.loc 1 791 0
	jle	.L725
	pxor	%xmm3, %xmm3
	xorl	%r11d, %r11d
	cvtsi2sd	%ecx, %xmm3
	addsd	.LC15(%rip), %xmm3
.LVL1544:
	.p2align 4,,10
	.p2align 3
.L726:
.LBB9319:
	.loc 1 794 0
	movzbl	(%rbx,%r11), %edx
	pxor	%xmm2, %xmm2
	.loc 1 795 0
	pxor	%xmm1, %xmm1
	leaq	(%r11,%r9), %rsi
	movl	$1, %edi
	addq	%rbx, %rsi
	.loc 1 794 0
	cvtsi2sd	%edx, %xmm2
.LVL1545:
	.loc 1 795 0
	movzbl	0(%rbp,%r11), %edx
	cvtsi2sd	%edx, %xmm1
.LVL1546:
	movq	%r11, %rdx
	subq	%r9, %rdx
	addq	%rbp, %rdx
.LVL1547:
	.p2align 4,,10
	.p2align 3
.L720:
.LBB9320:
	.loc 1 797 0 discriminator 2
	movzbl	(%rsi), %r8d
	pxor	%xmm0, %xmm0
	.loc 1 796 0 discriminator 2
	addl	$1, %edi
.LVL1548:
	addq	%r9, %rsi
	.loc 1 797 0 discriminator 2
	cvtsi2sd	%r8d, %xmm0
	.loc 1 798 0 discriminator 2
	movzbl	(%rdx), %r8d
	subq	%r9, %rdx
	.loc 1 796 0 discriminator 2
	cmpl	%r10d, %edi
	.loc 1 797 0 discriminator 2
	addsd	%xmm0, %xmm2
.LVL1549:
	.loc 1 798 0 discriminator 2
	pxor	%xmm0, %xmm0
	cvtsi2sd	%r8d, %xmm0
	addsd	%xmm0, %xmm1
.LVL1550:
	.loc 1 796 0 discriminator 2
	jne	.L720
.LVL1551:
.LBE9320:
.LBB9321:
.LBB9322:
.LBB9323:
.LBB9324:
.LBB9325:
.LBB9326:
	.loc 3 63 0
	divsd	%xmm3, %xmm2
.LVL1552:
.LBE9326:
.LBE9325:
.LBB9327:
.LBB9328:
	.loc 3 827 0
	cvtsd2si	%xmm2, %edx
.LVL1553:
.LBE9328:
.LBE9327:
.LBE9324:
.LBE9323:
.LBB9329:
.LBB9330:
	.loc 13 132 0
	cmpl	$255, %edx
	movl	%edx, %esi
	jbe	.L722
	testl	%edx, %edx
	setg	%sil
	negl	%esi
.L722:
.LVL1554:
.LBE9330:
.LBE9329:
.LBE9322:
.LBE9321:
.LBB9331:
.LBB9332:
.LBB9333:
.LBB9334:
.LBB9335:
.LBB9336:
	.loc 3 63 0
	divsd	%xmm3, %xmm1
.LVL1555:
.LBE9336:
.LBE9335:
.LBE9334:
.LBE9333:
.LBE9332:
.LBE9331:
	.loc 1 800 0
	movb	%sil, (%r12,%r11)
.LBB9344:
.LBB9343:
.LBB9340:
.LBB9339:
.LBB9337:
.LBB9338:
	.loc 3 827 0
	cvtsd2si	%xmm1, %edx
.LVL1556:
.LBE9338:
.LBE9337:
.LBE9339:
.LBE9340:
.LBB9341:
.LBB9342:
	.loc 13 132 0
	cmpl	$255, %edx
	movl	%edx, %esi
	jbe	.L724
	testl	%edx, %edx
	setg	%sil
	negl	%esi
.L724:
.LVL1557:
.LBE9342:
.LBE9341:
.LBE9343:
.LBE9344:
	.loc 1 801 0
	movb	%sil, 0(%r13,%r11)
.LVL1558:
	addq	$1, %r11
.LVL1559:
.LBE9319:
	.loc 1 791 0
	cmpl	%r11d, %ecx
	jg	.L726
.LVL1560:
.L725:
	addq	%rax, %r14
.LBE9345:
.LBE9317:
	.loc 1 786 0
	cmpl	32(%rsp), %r15d
	jge	.L718
	movq	16(%rsp), %rbx
	addl	$1, %r15d
.LVL1561:
	movq	16(%rbx), %rdx
	jmp	.L727
.LVL1562:
.L718:
.LBE9316:
.LBB9346:
.LBB9347:
.LBB9348:
.LBB9349:
.LBB9350:
	.loc 2 366 0
	movq	184(%rsp), %rax
	testq	%rax, %rax
	je	.L735
	lock subl	$1, (%rax)
	je	.L728
.LVL1563:
.L735:
.LBB9351:
	.loc 2 369 0
	movl	164(%rsp), %eax
.LBE9351:
	.loc 2 368 0
	movq	$0, 208(%rsp)
	movq	$0, 200(%rsp)
	movq	$0, 192(%rsp)
	movq	$0, 176(%rsp)
.LVL1564:
.LBB9352:
	.loc 2 369 0
	testl	%eax, %eax
	jle	.L732
	movq	224(%rsp), %rdx
	xorl	%eax, %eax
.LVL1565:
.L733:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL1566:
	addq	$4, %rdx
	cmpl	%eax, 164(%rsp)
	jg	.L733
.LVL1567:
.L732:
.LBE9352:
.LBE9350:
.LBE9349:
	.loc 2 277 0
	movq	232(%rsp), %rdi
	leaq	240(%rsp), %rax
.LBB9355:
.LBB9353:
	.loc 2 371 0
	movq	$0, 184(%rsp)
.LVL1568:
.LBE9353:
.LBE9355:
	.loc 2 277 0
	cmpq	%rax, %rdi
	je	.L657
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL1569:
.L657:
.LBE9348:
.LBE9347:
.LBE9346:
	.loc 1 805 0
	movq	264(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L773
	addq	$280, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.LVL1570:
.L741:
	.cfi_restore_state
.LBB9359:
.LBB9233:
.LBB9226:
.LBB9224:
	.loc 2 910 0
	movq	%rsi, %r9
	movq	%rdi, %r10
.LBE9224:
.LBE9226:
.LBE9233:
.LBE9359:
.LBB9360:
	.loc 1 732 0
	xorl	%r11d, %r11d
	jmp	.L666
.L746:
.LBE9360:
.LBB9361:
.LBB9234:
.LBB9227:
.LBB9225:
	.loc 2 910 0
	xorl	%eax, %eax
.LVL1571:
.L665:
.LBE9225:
.LBE9227:
.LBE9234:
.LBE9361:
.LBB9362:
	.loc 1 733 0
	movzbl	(%rsi,%rax), %edx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%edx, %xmm0
	movsd	%xmm0, (%rdi,%rax,8)
.LVL1572:
	addq	$1, %rax
.LVL1573:
	.loc 1 732 0
	cmpl	%eax, %r15d
	jg	.L665
	jmp	.L664
.LVL1574:
.L745:
.LBE9362:
.LBB9363:
.LBB9235:
	.loc 2 810 0
	leaq	160(%rsp), %rdi
.LVL1575:
	movq	%rax, %rbx
.LVL1576:
	call	_ZN2cv3MatD2Ev
.LVL1577:
	movq	%rbx, %rdi
.LEHB2:
	call	_Unwind_Resume
.LVL1578:
.LEHE2:
.L771:
.LBE9235:
.LBE9363:
.LBB9364:
	.loc 1 768 0
	testl	%ebx, %ebx
	jle	.L718
	movq	16(%rsp), %rax
	movl	%r14d, %edi
	movq	16(%rax), %rdx
.LVL1579:
	movslq	%ebp, %rax
	jmp	.L738
.L744:
.LVL1580:
.L728:
.LBE9364:
.LBB9365:
.LBB9358:
.LBB9357:
.LBB9356:
.LBB9354:
	.loc 2 367 0
	leaq	160(%rsp), %rdi
.LVL1581:
	call	_ZN2cv3Mat10deallocateEv
.LVL1582:
	jmp	.L735
.LVL1583:
.L742:
.LBE9354:
.LBE9356:
.LBE9357:
.LBE9358:
.LBE9365:
.LBB9366:
	.loc 1 735 0
	movl	%r15d, 32(%rsp)
	movq	%r9, 40(%rsp)
	movq	%rdx, 48(%rsp)
	jmp	.L676
.L675:
	movl	%ebp, %eax
	addq	%rdx, %r10
	subl	%r13d, %eax
	leal	-2(%rax), %r8d
	xorl	%eax, %eax
	addq	$1, %r8
.LVL1584:
.L683:
	.loc 1 736 0
	movzbl	(%r9,%rax), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2sd	%ecx, %xmm0
	addsd	(%r10,%rax,8), %xmm0
	movsd	%xmm0, (%rdx,%rax,8)
.LVL1585:
	addq	$1, %rax
.LVL1586:
	.loc 1 735 0
	cmpq	%rax, %r8
	jne	.L683
	jmp	.L682
.LVL1587:
.L773:
.LBE9366:
	.loc 1 805 0
	call	__stack_chk_fail
.LVL1588:
	.cfi_endproc
.LFE11288:
	.section	.gcc_except_table
.LLSDA11288:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE11288-.LLSDACSB11288
.LLSDACSB11288:
	.uleb128 .LEHB0-.LFB11288
	.uleb128 .LEHE0-.LEHB0
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB1-.LFB11288
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L745-.LFB11288
	.uleb128 0
	.uleb128 .LEHB2-.LFB11288
	.uleb128 .LEHE2-.LEHB2
	.uleb128 0
	.uleb128 0
.LLSDACSE11288:
	.text
	.size	_Z9boxFilterRKN2cv3MatERS0_ii, .-_Z9boxFilterRKN2cv3MatERS0_ii
	.section	.text.unlikely
.LCOLDE48:
	.text
.LHOTE48:
	.section	.text.unlikely
	.align 2
.LCOLDB49:
	.text
.LHOTB49:
	.align 2
	.p2align 4,,15
	.globl	_ZN15BilateralFilterC2Edddd
	.type	_ZN15BilateralFilterC2Edddd, @function
_ZN15BilateralFilterC2Edddd:
.LFB11296:
	.loc 1 822 0
	.cfi_startproc
.LVL1589:
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	movq	%rdi, %r12
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
.LBB9367:
.LBB9368:
	.loc 1 824 0
	xorl	%ebp, %ebp
	xorl	%ebx, %ebx
.LBE9368:
.LBE9367:
	.loc 1 822 0
	subq	$16, %rsp
	.cfi_def_cfa_offset 48
.LBB9373:
	.loc 1 822 0
	movsd	%xmm0, 8(%rdi)
.LBB9370:
	.loc 1 824 0
	movapd	%xmm2, %xmm0
.LVL1590:
.LBE9370:
	.loc 1 822 0
	movsd	%xmm2, 24(%rdi)
.LVL1591:
	movsd	%xmm3, (%rdi)
.LBB9371:
	.loc 1 824 0
	addsd	%xmm2, %xmm0
.LBE9371:
	.loc 1 822 0
	movsd	%xmm1, 16(%rdi)
.LBB9372:
	.loc 1 824 0
	mulsd	%xmm0, %xmm2
.LVL1592:
	movsd	.LC15(%rip), %xmm0
	divsd	%xmm2, %xmm0
	movsd	%xmm0, 8(%rsp)
.LVL1593:
	.p2align 4,,10
	.p2align 3
.L775:
.LBB9369:
	.loc 1 826 0 discriminator 2
	movl	%ebp, %eax
	pxor	%xmm0, %xmm0
	imull	%ebx, %eax
	subl	$1, %ebp
	cvtsi2sd	%eax, %xmm0
	mulsd	8(%rsp), %xmm0
	call	exp
.LVL1594:
	movsd	%xmm0, 32(%r12,%rbx,8)
.LVL1595:
	addq	$1, %rbx
.LVL1596:
	.loc 1 825 0 discriminator 2
	cmpq	$255, %rbx
	jne	.L775
.LBE9369:
.LBE9372:
.LBE9373:
	.loc 1 829 0
	addq	$16, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
.LVL1597:
	ret
	.cfi_endproc
.LFE11296:
	.size	_ZN15BilateralFilterC2Edddd, .-_ZN15BilateralFilterC2Edddd
	.section	.text.unlikely
.LCOLDE49:
	.text
.LHOTE49:
	.globl	_ZN15BilateralFilterC1Edddd
	.set	_ZN15BilateralFilterC1Edddd,_ZN15BilateralFilterC2Edddd
	.section	.text.unlikely._ZN2cv7MatExprD2Ev,"axG",@progbits,_ZN2cv7MatExprD5Ev,comdat
	.align 2
.LCOLDB50:
	.section	.text._ZN2cv7MatExprD2Ev,"axG",@progbits,_ZN2cv7MatExprD5Ev,comdat
.LHOTB50:
	.align 2
	.p2align 4,,15
	.weak	_ZN2cv7MatExprD2Ev
	.type	_ZN2cv7MatExprD2Ev, @function
_ZN2cv7MatExprD2Ev:
.LFB11301:
	.loc 2 1219 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA11301
.LVL1598:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
.LBB9393:
.LBB9394:
.LBB9395:
.LBB9396:
.LBB9397:
	.loc 2 366 0
	movq	232(%rdi), %rax
.LBE9397:
.LBE9396:
.LBE9395:
.LBE9394:
.LBE9393:
	.loc 2 1219 0
	movq	%rdi, %rbx
.LBB9436:
.LBB9408:
.LBB9406:
.LBB9403:
.LBB9400:
	.loc 2 366 0
	testq	%rax, %rax
	je	.L800
	lock subl	$1, (%rax)
	jne	.L800
.LBE9400:
.LBE9403:
.LBE9406:
.LBE9408:
	.loc 2 1219 0
	leaq	208(%rdi), %rdi
.LVL1599:
.LBB9409:
.LBB9407:
.LBB9404:
.LBB9401:
	.loc 2 367 0
	call	_ZN2cv3Mat10deallocateEv
.LVL1600:
.L800:
.LBB9398:
	.loc 2 369 0
	movl	212(%rbx), %ecx
.LBE9398:
	.loc 2 368 0
	movq	$0, 256(%rbx)
	movq	$0, 248(%rbx)
	movq	$0, 240(%rbx)
	movq	$0, 224(%rbx)
.LVL1601:
.LBB9399:
	.loc 2 369 0
	testl	%ecx, %ecx
	jle	.L784
	movq	272(%rbx), %rdx
	xorl	%eax, %eax
.LVL1602:
	.p2align 4,,10
	.p2align 3
.L785:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL1603:
	addq	$4, %rdx
	cmpl	%eax, 212(%rbx)
	jg	.L785
.LVL1604:
.L784:
.LBE9399:
.LBE9401:
.LBE9404:
	.loc 2 277 0
	movq	280(%rbx), %rdi
	leaq	288(%rbx), %rax
.LBB9405:
.LBB9402:
	.loc 2 371 0
	movq	$0, 232(%rbx)
.LVL1605:
.LBE9402:
.LBE9405:
	.loc 2 277 0
	cmpq	%rax, %rdi
	je	.L783
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL1606:
.L783:
.LBE9407:
.LBE9409:
.LBB9410:
.LBB9411:
.LBB9412:
.LBB9413:
	.loc 2 366 0
	movq	136(%rbx), %rax
	testq	%rax, %rax
	je	.L801
	lock subl	$1, (%rax)
	jne	.L801
.LBE9413:
.LBE9412:
.LBE9411:
.LBE9410:
	.loc 2 1219 0
	leaq	112(%rbx), %rdi
.LVL1607:
.LBB9421:
.LBB9420:
.LBB9418:
.LBB9416:
	.loc 2 367 0
	call	_ZN2cv3Mat10deallocateEv
.LVL1608:
.L801:
.LBB9414:
	.loc 2 369 0
	movl	116(%rbx), %edx
.LBE9414:
	.loc 2 368 0
	movq	$0, 160(%rbx)
	movq	$0, 152(%rbx)
	movq	$0, 144(%rbx)
	movq	$0, 128(%rbx)
.LVL1609:
.LBB9415:
	.loc 2 369 0
	testl	%edx, %edx
	jle	.L791
	movq	176(%rbx), %rdx
	xorl	%eax, %eax
.LVL1610:
	.p2align 4,,10
	.p2align 3
.L792:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL1611:
	addq	$4, %rdx
	cmpl	%eax, 116(%rbx)
	jg	.L792
.LVL1612:
.L791:
.LBE9415:
.LBE9416:
.LBE9418:
	.loc 2 277 0
	movq	184(%rbx), %rdi
	leaq	192(%rbx), %rax
.LBB9419:
.LBB9417:
	.loc 2 371 0
	movq	$0, 136(%rbx)
.LVL1613:
.LBE9417:
.LBE9419:
	.loc 2 277 0
	cmpq	%rax, %rdi
	je	.L790
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL1614:
.L790:
.LBE9420:
.LBE9421:
.LBB9422:
.LBB9423:
.LBB9424:
.LBB9425:
	.loc 2 366 0
	movq	40(%rbx), %rax
	testq	%rax, %rax
	je	.L802
	lock subl	$1, (%rax)
	jne	.L802
.LBE9425:
.LBE9424:
.LBE9423:
.LBE9422:
	.loc 2 1219 0
	leaq	16(%rbx), %rdi
.LVL1615:
.LBB9434:
.LBB9432:
.LBB9430:
.LBB9428:
	.loc 2 367 0
	call	_ZN2cv3Mat10deallocateEv
.LVL1616:
.L802:
.LBB9426:
	.loc 2 369 0
	movl	20(%rbx), %eax
.LBE9426:
	.loc 2 368 0
	movq	$0, 64(%rbx)
	movq	$0, 56(%rbx)
	movq	$0, 48(%rbx)
	movq	$0, 32(%rbx)
.LVL1617:
.LBB9427:
	.loc 2 369 0
	testl	%eax, %eax
	jle	.L798
	movq	80(%rbx), %rdx
	xorl	%eax, %eax
.LVL1618:
	.p2align 4,,10
	.p2align 3
.L799:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL1619:
	addq	$4, %rdx
	cmpl	%eax, 20(%rbx)
	jg	.L799
.LVL1620:
.L798:
.LBE9427:
.LBE9428:
.LBE9430:
	.loc 2 277 0
	movq	88(%rbx), %rdi
.LBB9431:
.LBB9429:
	.loc 2 371 0
	movq	$0, 40(%rbx)
.LVL1621:
.LBE9429:
.LBE9431:
	.loc 2 277 0
	addq	$96, %rbx
.LVL1622:
	cmpq	%rbx, %rdi
	je	.L815
.LBE9432:
.LBE9434:
.LBE9436:
	.loc 2 1219 0
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 8
.LVL1623:
.LBB9437:
.LBB9435:
.LBB9433:
	.loc 2 278 0
	jmp	_ZN2cv8fastFreeEPv
.LVL1624:
	.p2align 4,,10
	.p2align 3
.L815:
	.cfi_restore_state
.LBE9433:
.LBE9435:
.LBE9437:
	.loc 2 1219 0
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE11301:
	.section	.gcc_except_table
.LLSDA11301:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE11301-.LLSDACSB11301
.LLSDACSB11301:
.LLSDACSE11301:
	.section	.text._ZN2cv7MatExprD2Ev,"axG",@progbits,_ZN2cv7MatExprD5Ev,comdat
	.size	_ZN2cv7MatExprD2Ev, .-_ZN2cv7MatExprD2Ev
	.section	.text.unlikely._ZN2cv7MatExprD2Ev,"axG",@progbits,_ZN2cv7MatExprD5Ev,comdat
.LCOLDE50:
	.section	.text._ZN2cv7MatExprD2Ev,"axG",@progbits,_ZN2cv7MatExprD5Ev,comdat
.LHOTE50:
	.weak	_ZN2cv7MatExprD1Ev
	.set	_ZN2cv7MatExprD1Ev,_ZN2cv7MatExprD2Ev
	.section	.text.unlikely._ZNSt6vectorIN2cv3MatESaIS1_EED2Ev,"axG",@progbits,_ZNSt6vectorIN2cv3MatESaIS1_EED5Ev,comdat
	.align 2
.LCOLDB51:
	.section	.text._ZNSt6vectorIN2cv3MatESaIS1_EED2Ev,"axG",@progbits,_ZNSt6vectorIN2cv3MatESaIS1_EED5Ev,comdat
.LHOTB51:
	.align 2
	.p2align 4,,15
	.weak	_ZNSt6vectorIN2cv3MatESaIS1_EED2Ev
	.type	_ZNSt6vectorIN2cv3MatESaIS1_EED2Ev, @function
_ZNSt6vectorIN2cv3MatESaIS1_EED2Ev:
.LFB11495:
	.file 15 "/usr/include/c++/5/bits/stl_vector.h"
	.loc 15 423 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA11495
.LVL1625:
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
.LBB9482:
	.loc 15 424 0
	movq	8(%rdi), %rbp
	movq	(%rdi), %rbx
.LVL1626:
.LBB9483:
.LBB9484:
.LBB9485:
.LBB9486:
.LBB9487:
	.file 16 "/usr/include/c++/5/bits/stl_construct.h"
	.loc 16 102 0
	cmpq	%rbx, %rbp
	je	.L817
	movq	%rdi, %r12
.LVL1627:
	.p2align 4,,10
	.p2align 3
.L825:
.LBB9488:
.LBB9489:
.LBB9490:
.LBB9491:
.LBB9492:
	.loc 2 366 0
	movq	24(%rbx), %rax
	testq	%rax, %rax
	je	.L827
	lock subl	$1, (%rax)
	jne	.L827
	.loc 2 367 0
	movq	%rbx, %rdi
	call	_ZN2cv3Mat10deallocateEv
.LVL1628:
.L827:
.LBB9493:
	.loc 2 369 0
	movl	4(%rbx), %eax
.LBE9493:
	.loc 2 368 0
	movq	$0, 48(%rbx)
	movq	$0, 40(%rbx)
	movq	$0, 32(%rbx)
	movq	$0, 16(%rbx)
.LVL1629:
.LBB9494:
	.loc 2 369 0
	testl	%eax, %eax
	jle	.L823
	movq	64(%rbx), %rdx
	xorl	%eax, %eax
.LVL1630:
	.p2align 4,,10
	.p2align 3
.L824:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL1631:
	addq	$4, %rdx
	cmpl	%eax, 4(%rbx)
	jg	.L824
.LVL1632:
.L823:
.LBE9494:
.LBE9492:
.LBE9491:
	.loc 2 277 0
	movq	72(%rbx), %rdi
	leaq	80(%rbx), %rax
.LBB9496:
.LBB9495:
	.loc 2 371 0
	movq	$0, 24(%rbx)
.LVL1633:
.LBE9495:
.LBE9496:
	.loc 2 277 0
	cmpq	%rax, %rdi
	je	.L822
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL1634:
.L822:
.LBE9490:
.LBE9489:
.LBE9488:
	.loc 16 102 0
	addq	$96, %rbx
.LVL1635:
	cmpq	%rbx, %rbp
	jne	.L825
	movq	(%r12), %rbp
.LVL1636:
.L817:
.LBE9487:
.LBE9486:
.LBE9485:
.LBE9484:
.LBE9483:
.LBB9497:
.LBB9498:
.LBB9499:
	.loc 15 177 0
	testq	%rbp, %rbp
	je	.L816
.LVL1637:
.LBE9499:
.LBE9498:
.LBE9497:
.LBE9482:
	.loc 15 425 0
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 24
.LBB9512:
.LBB9510:
.LBB9508:
.LBB9506:
.LBB9500:
.LBB9501:
.LBB9502:
	.file 17 "/usr/include/c++/5/ext/new_allocator.h"
	.loc 17 110 0
	movq	%rbp, %rdi
.LBE9502:
.LBE9501:
.LBE9500:
.LBE9506:
.LBE9508:
.LBE9510:
.LBE9512:
	.loc 15 425 0
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
.LBB9513:
.LBB9511:
.LBB9509:
.LBB9507:
.LBB9505:
.LBB9504:
.LBB9503:
	.loc 17 110 0
	jmp	_ZdlPv
.LVL1638:
.L816:
	.cfi_restore_state
.LBE9503:
.LBE9504:
.LBE9505:
.LBE9507:
.LBE9509:
.LBE9511:
.LBE9513:
	.loc 15 425 0
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE11495:
	.section	.gcc_except_table
.LLSDA11495:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE11495-.LLSDACSB11495
.LLSDACSB11495:
.LLSDACSE11495:
	.section	.text._ZNSt6vectorIN2cv3MatESaIS1_EED2Ev,"axG",@progbits,_ZNSt6vectorIN2cv3MatESaIS1_EED5Ev,comdat
	.size	_ZNSt6vectorIN2cv3MatESaIS1_EED2Ev, .-_ZNSt6vectorIN2cv3MatESaIS1_EED2Ev
	.section	.text.unlikely._ZNSt6vectorIN2cv3MatESaIS1_EED2Ev,"axG",@progbits,_ZNSt6vectorIN2cv3MatESaIS1_EED5Ev,comdat
.LCOLDE51:
	.section	.text._ZNSt6vectorIN2cv3MatESaIS1_EED2Ev,"axG",@progbits,_ZNSt6vectorIN2cv3MatESaIS1_EED5Ev,comdat
.LHOTE51:
	.weak	_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev
	.set	_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev,_ZNSt6vectorIN2cv3MatESaIS1_EED2Ev
	.section	.text.unlikely
	.align 2
.LCOLDB52:
	.text
.LHOTB52:
	.align 2
	.p2align 4,,15
	.globl	_ZN15BilateralFilter5applyERKN2cv3MatERS1_
	.type	_ZN15BilateralFilter5applyERKN2cv3MatERS1_, @function
_ZN15BilateralFilter5applyERKN2cv3MatERS1_:
.LFB11298:
	.loc 1 832 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA11298
.LVL1639:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	movq	%rdi, %r15
	pushq	%rbx
	movq	%rsi, %r13
	subq	$1864, %rsp
	.cfi_offset 3, -56
	.loc 1 832 0
	movq	%rdx, -1904(%rbp)
.LBB10066:
.LBB10067:
.LBB10068:
.LBB10069:
	.loc 15 185 0
	movq	$0, -1880(%rbp)
.LBE10069:
.LBE10068:
.LBE10067:
.LBE10066:
	.loc 1 833 0
	movsd	.LC6(%rip), %xmm0
	.loc 1 832 0
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
	.loc 1 833 0
	divsd	(%rdi), %xmm0
	cvttsd2si	%xmm0, %eax
	addl	$1, %eax
	.loc 1 834 0
	movslq	%eax, %rbx
	.loc 1 833 0
	movl	%eax, -1780(%rbp)
.LVL1640:
.LBB10108:
.LBB10084:
.LBB10082:
.LBB10080:
.LBB10070:
.LBB10071:
	.loc 15 170 0
	testq	%rbx, %rbx
	je	.L834
.LVL1641:
.LBB10072:
.LBB10073:
.LBB10074:
	.loc 17 104 0
	movq	%rbx, %rdi
.LVL1642:
.LEHB3:
	call	_Znwm
.LVL1643:
.LEHE3:
.LBE10074:
.LBE10073:
.LBE10072:
.LBE10071:
.LBE10070:
.LBE10080:
.LBE10082:
.LBE10084:
.LBB10085:
.LBB10086:
.LBB10087:
.LBB10088:
.LBB10089:
.LBB10090:
.LBB10091:
.LBB10092:
.LBB10093:
.LBB10094:
.LBB10095:
	.loc 6 723 0
	movq	%rbx, %rdx
	xorl	%esi, %esi
	movq	%rax, %rdi
.LBE10095:
.LBE10094:
.LBE10093:
.LBE10092:
.LBE10091:
.LBE10090:
.LBE10089:
.LBE10088:
.LBE10087:
.LBE10086:
.LBE10085:
.LBB10106:
.LBB10083:
.LBB10081:
.LBB10079:
.LBB10078:
.LBB10077:
.LBB10076:
.LBB10075:
	.loc 17 104 0
	movq	%rax, -1880(%rbp)
.LVL1644:
.LBE10075:
.LBE10076:
.LBE10077:
.LBE10078:
.LBE10079:
.LBE10081:
.LBE10083:
.LBE10106:
.LBB10107:
.LBB10105:
.LBB10104:
.LBB10103:
.LBB10102:
.LBB10101:
.LBB10100:
.LBB10099:
.LBB10098:
.LBB10097:
.LBB10096:
	.loc 6 723 0
	call	memset
.LVL1645:
.L834:
.LBE10096:
.LBE10097:
.LBE10098:
.LBE10099:
.LBE10100:
.LBE10101:
.LBE10102:
.LBE10103:
.LBE10104:
.LBE10105:
.LBE10107:
.LBE10108:
.LBB10109:
.LBB10110:
.LBB10111:
	.loc 2 709 0
	leaq	-1168(%rbp), %rsi
.LBE10111:
.LBE10110:
.LBE10109:
.LBB10134:
.LBB10135:
	.loc 2 713 0
	movq	64(%r13), %rax
.LBE10135:
.LBE10134:
.LBB10137:
.LBB10114:
.LBB10115:
	.loc 2 353 0
	leaq	-1168(%rbp), %rdi
.LBE10115:
.LBE10114:
.LBB10120:
.LBB10112:
	.loc 2 709 0
	leaq	8(%rsi), %rcx
.LBE10112:
.LBE10120:
.LBE10137:
.LBB10138:
.LBB10136:
	.loc 2 713 0
	movl	(%rax), %edx
	movl	4(%rax), %eax
.LVL1646:
.LBE10136:
.LBE10138:
.LBB10139:
.LBB10121:
.LBB10113:
	.loc 2 709 0
	movq	%rcx, -1104(%rbp)
.LVL1647:
.LBE10113:
.LBE10121:
.LBB10122:
.LBB10123:
	.loc 2 738 0
	leaq	80(%rsi), %rcx
.LBE10123:
.LBE10122:
.LBB10125:
.LBB10116:
	.loc 2 353 0
	movl	$2, %esi
.LVL1648:
.LBE10116:
.LBE10125:
.LBB10126:
.LBB10124:
	.loc 2 738 0
	movq	$0, -1080(%rbp)
	movq	$0, -1088(%rbp)
.LVL1649:
	movq	%rcx, -1096(%rbp)
.LBE10124:
.LBE10126:
.LBB10127:
.LBB10117:
	.loc 2 353 0
	movl	0(%r13), %ecx
	.loc 2 352 0
	movl	%edx, -1664(%rbp)
	.loc 2 353 0
	leaq	-1664(%rbp), %rdx
.LBE10117:
.LBE10127:
.LBB10128:
.LBB10129:
	.loc 2 60 0
	movl	$1124007936, -1168(%rbp)
	.loc 2 61 0
	movl	$0, -1156(%rbp)
	movl	$0, -1160(%rbp)
.LBE10129:
.LBE10128:
.LBB10131:
.LBB10118:
	.loc 2 353 0
	andl	$4095, %ecx
.LVL1650:
.LBE10118:
.LBE10131:
.LBB10132:
.LBB10130:
	.loc 2 61 0
	movl	$0, -1164(%rbp)
	.loc 2 62 0
	movq	$0, -1120(%rbp)
	movq	$0, -1128(%rbp)
	movq	$0, -1136(%rbp)
	movq	$0, -1152(%rbp)
	.loc 2 63 0
	movq	$0, -1144(%rbp)
	.loc 2 64 0
	movq	$0, -1112(%rbp)
.LVL1651:
.LBE10130:
.LBE10132:
.LBB10133:
.LBB10119:
	.loc 2 352 0
	movl	%eax, -1660(%rbp)
.LEHB4:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL1652:
.LEHE4:
.LBE10119:
.LBE10133:
.LBE10139:
.LBB10140:
.LBB10141:
.LBB10142:
	.loc 2 709 0
	leaq	-1072(%rbp), %rsi
.LBE10142:
.LBE10141:
.LBE10140:
.LBB10165:
.LBB10166:
	.loc 2 713 0
	movq	64(%r13), %rax
.LBE10166:
.LBE10165:
.LBB10168:
.LBB10145:
.LBB10146:
	.loc 2 353 0
	leaq	-1072(%rbp), %rdi
.LBE10146:
.LBE10145:
.LBB10151:
.LBB10143:
	.loc 2 709 0
	leaq	8(%rsi), %rcx
.LBE10143:
.LBE10151:
.LBE10168:
.LBB10169:
.LBB10167:
	.loc 2 713 0
	movl	(%rax), %edx
	movl	4(%rax), %eax
.LVL1653:
.LBE10167:
.LBE10169:
.LBB10170:
.LBB10152:
.LBB10144:
	.loc 2 709 0
	movq	%rcx, -1008(%rbp)
.LVL1654:
.LBE10144:
.LBE10152:
.LBB10153:
.LBB10154:
	.loc 2 738 0
	leaq	80(%rsi), %rcx
.LBE10154:
.LBE10153:
.LBB10156:
.LBB10147:
	.loc 2 353 0
	movl	$2, %esi
.LVL1655:
.LBE10147:
.LBE10156:
.LBB10157:
.LBB10155:
	.loc 2 738 0
	movq	$0, -984(%rbp)
	movq	$0, -992(%rbp)
.LVL1656:
	movq	%rcx, -1000(%rbp)
.LBE10155:
.LBE10157:
.LBB10158:
.LBB10148:
	.loc 2 353 0
	movl	0(%r13), %ecx
	.loc 2 352 0
	movl	%edx, -1648(%rbp)
	.loc 2 353 0
	leaq	-1648(%rbp), %rdx
.LBE10148:
.LBE10158:
.LBB10159:
.LBB10160:
	.loc 2 60 0
	movl	$1124007936, -1072(%rbp)
	.loc 2 61 0
	movl	$0, -1060(%rbp)
	movl	$0, -1064(%rbp)
.LBE10160:
.LBE10159:
.LBB10162:
.LBB10149:
	.loc 2 353 0
	andl	$4095, %ecx
.LVL1657:
.LBE10149:
.LBE10162:
.LBB10163:
.LBB10161:
	.loc 2 61 0
	movl	$0, -1068(%rbp)
	.loc 2 62 0
	movq	$0, -1024(%rbp)
	movq	$0, -1032(%rbp)
	movq	$0, -1040(%rbp)
	movq	$0, -1056(%rbp)
	.loc 2 63 0
	movq	$0, -1048(%rbp)
	.loc 2 64 0
	movq	$0, -1016(%rbp)
.LVL1658:
.LBE10161:
.LBE10163:
.LBB10164:
.LBB10150:
	.loc 2 352 0
	movl	%eax, -1644(%rbp)
.LEHB5:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL1659:
.LEHE5:
.LBE10150:
.LBE10164:
.LBE10170:
.LBB10171:
.LBB10172:
.LBB10173:
	.loc 2 709 0
	leaq	-976(%rbp), %rsi
.LBE10173:
.LBE10172:
.LBE10171:
.LBB10196:
.LBB10197:
	.loc 2 713 0
	movq	64(%r13), %rax
.LBE10197:
.LBE10196:
.LBB10199:
.LBB10176:
.LBB10177:
	.loc 2 353 0
	leaq	-976(%rbp), %rdi
.LBE10177:
.LBE10176:
.LBB10182:
.LBB10174:
	.loc 2 709 0
	leaq	8(%rsi), %rcx
.LBE10174:
.LBE10182:
.LBE10199:
.LBB10200:
.LBB10198:
	.loc 2 713 0
	movl	(%rax), %edx
	movl	4(%rax), %eax
.LVL1660:
.LBE10198:
.LBE10200:
.LBB10201:
.LBB10183:
.LBB10175:
	.loc 2 709 0
	movq	%rcx, -912(%rbp)
.LVL1661:
.LBE10175:
.LBE10183:
.LBB10184:
.LBB10185:
	.loc 2 738 0
	leaq	80(%rsi), %rcx
.LBE10185:
.LBE10184:
.LBB10187:
.LBB10178:
	.loc 2 353 0
	movl	$2, %esi
.LVL1662:
.LBE10178:
.LBE10187:
.LBB10188:
.LBB10186:
	.loc 2 738 0
	movq	$0, -888(%rbp)
	movq	$0, -896(%rbp)
.LVL1663:
	movq	%rcx, -904(%rbp)
.LBE10186:
.LBE10188:
.LBB10189:
.LBB10179:
	.loc 2 353 0
	movl	0(%r13), %ecx
	.loc 2 352 0
	movl	%edx, -1632(%rbp)
	.loc 2 353 0
	leaq	-1632(%rbp), %rdx
.LBE10179:
.LBE10189:
.LBB10190:
.LBB10191:
	.loc 2 60 0
	movl	$1124007936, -976(%rbp)
	.loc 2 61 0
	movl	$0, -964(%rbp)
	movl	$0, -968(%rbp)
.LBE10191:
.LBE10190:
.LBB10193:
.LBB10180:
	.loc 2 353 0
	andl	$4095, %ecx
.LVL1664:
.LBE10180:
.LBE10193:
.LBB10194:
.LBB10192:
	.loc 2 61 0
	movl	$0, -972(%rbp)
	.loc 2 62 0
	movq	$0, -928(%rbp)
	movq	$0, -936(%rbp)
	movq	$0, -944(%rbp)
	movq	$0, -960(%rbp)
	.loc 2 63 0
	movq	$0, -952(%rbp)
	.loc 2 64 0
	movq	$0, -920(%rbp)
.LVL1665:
.LBE10192:
.LBE10194:
.LBB10195:
.LBB10181:
	.loc 2 352 0
	movl	%eax, -1628(%rbp)
.LEHB6:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL1666:
.LEHE6:
.LBE10181:
.LBE10195:
.LBE10201:
.LBB10202:
.LBB10203:
.LBB10204:
	.loc 2 709 0
	leaq	-880(%rbp), %rsi
.LBE10204:
.LBE10203:
.LBE10202:
.LBB10227:
.LBB10228:
	.loc 2 713 0
	movq	64(%r13), %rax
.LBE10228:
.LBE10227:
.LBB10230:
.LBB10207:
.LBB10208:
	.loc 2 353 0
	leaq	-880(%rbp), %rdi
.LBE10208:
.LBE10207:
.LBB10213:
.LBB10205:
	.loc 2 709 0
	leaq	8(%rsi), %rcx
.LBE10205:
.LBE10213:
.LBE10230:
.LBB10231:
.LBB10229:
	.loc 2 713 0
	movl	(%rax), %edx
	movl	4(%rax), %eax
.LVL1667:
.LBE10229:
.LBE10231:
.LBB10232:
.LBB10214:
.LBB10206:
	.loc 2 709 0
	movq	%rcx, -816(%rbp)
.LVL1668:
.LBE10206:
.LBE10214:
.LBB10215:
.LBB10216:
	.loc 2 738 0
	leaq	80(%rsi), %rcx
.LBE10216:
.LBE10215:
.LBB10218:
.LBB10209:
	.loc 2 353 0
	movl	$2, %esi
.LVL1669:
.LBE10209:
.LBE10218:
.LBB10219:
.LBB10217:
	.loc 2 738 0
	movq	$0, -792(%rbp)
	movq	$0, -800(%rbp)
.LVL1670:
	movq	%rcx, -808(%rbp)
.LBE10217:
.LBE10219:
.LBB10220:
.LBB10210:
	.loc 2 353 0
	movl	0(%r13), %ecx
	.loc 2 352 0
	movl	%edx, -1616(%rbp)
	.loc 2 353 0
	leaq	-1616(%rbp), %rdx
.LBE10210:
.LBE10220:
.LBB10221:
.LBB10222:
	.loc 2 60 0
	movl	$1124007936, -880(%rbp)
	.loc 2 61 0
	movl	$0, -868(%rbp)
	movl	$0, -872(%rbp)
.LBE10222:
.LBE10221:
.LBB10224:
.LBB10211:
	.loc 2 353 0
	andl	$4095, %ecx
.LVL1671:
.LBE10211:
.LBE10224:
.LBB10225:
.LBB10223:
	.loc 2 61 0
	movl	$0, -876(%rbp)
	.loc 2 62 0
	movq	$0, -832(%rbp)
	movq	$0, -840(%rbp)
	movq	$0, -848(%rbp)
	movq	$0, -864(%rbp)
	.loc 2 63 0
	movq	$0, -856(%rbp)
	.loc 2 64 0
	movq	$0, -824(%rbp)
.LVL1672:
.LBE10223:
.LBE10225:
.LBB10226:
.LBB10212:
	.loc 2 352 0
	movl	%eax, -1612(%rbp)
.LEHB7:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL1673:
.LEHE7:
.LBE10212:
.LBE10226:
.LBE10232:
.LBB10233:
.LBB10234:
.LBB10235:
.LBB10236:
.LBB10237:
.LBB10238:
	.loc 15 170 0
	testq	%rbx, %rbx
.LBE10238:
.LBE10237:
.LBE10236:
.LBE10235:
.LBB10254:
.LBB10255:
	.loc 15 91 0
	movq	$0, -1744(%rbp)
	movq	$0, -1736(%rbp)
	movq	$0, -1728(%rbp)
.LVL1674:
.LBE10255:
.LBE10254:
.LBB10256:
.LBB10251:
.LBB10248:
.LBB10245:
	.loc 15 170 0
	je	.L835
.LVL1675:
.LBB10239:
.LBB10240:
.LBB10241:
	.loc 17 101 0
	movabsq	$192153584101141162, %rax
	cmpq	%rax, %rbx
	ja	.L1021
	.loc 17 104 0
	leaq	(%rbx,%rbx,2), %rax
	salq	$5, %rax
	movq	%rax, %rdi
	movq	%rax, %r14
.LEHB8:
	call	_Znwm
.LVL1676:
.LEHE8:
.LBE10241:
.LBE10240:
.LBE10239:
.LBE10245:
.LBE10248:
	.loc 15 187 0
	movq	%r14, %rsi
	.loc 15 185 0
	movq	%rax, -1744(%rbp)
	.loc 15 187 0
	addq	%rax, %rsi
	movq	%rsi, -1728(%rbp)
.LVL1677:
	.p2align 4,,10
	.p2align 3
.L838:
.LBE10251:
.LBE10256:
.LBE10234:
.LBB10259:
.LBB10260:
.LBB10261:
.LBB10262:
.LBB10263:
.LBB10264:
.LBB10265:
.LBB10266:
.LBB10267:
	.loc 16 75 0
	testq	%rax, %rax
	je	.L837
.LVL1678:
.LBB10268:
.LBB10269:
.LBB10270:
	.loc 2 709 0
	leaq	8(%rax), %rdx
.LVL1679:
.LBE10270:
.LBE10269:
.LBB10272:
.LBB10273:
	.loc 2 738 0
	movq	$0, 88(%rax)
	movq	$0, 80(%rax)
.LBE10273:
.LBE10272:
.LBB10276:
.LBB10277:
	.loc 2 60 0
	movl	$1124007936, (%rax)
	.loc 2 61 0
	movl	$0, 12(%rax)
.LBE10277:
.LBE10276:
.LBB10280:
.LBB10271:
	.loc 2 709 0
	movq	%rdx, 64(%rax)
.LVL1680:
.LBE10271:
.LBE10280:
.LBB10281:
.LBB10274:
	.loc 2 738 0
	leaq	80(%rax), %rdx
.LBE10274:
.LBE10281:
.LBB10282:
.LBB10278:
	.loc 2 61 0
	movl	$0, 8(%rax)
	movl	$0, 4(%rax)
	.loc 2 62 0
	movq	$0, 48(%rax)
.LBE10278:
.LBE10282:
.LBB10283:
.LBB10275:
	.loc 2 738 0
	movq	%rdx, 72(%rax)
.LBE10275:
.LBE10283:
.LBB10284:
.LBB10279:
	.loc 2 62 0
	movq	$0, 40(%rax)
	movq	$0, 32(%rax)
	movq	$0, 16(%rax)
	.loc 2 63 0
	movq	$0, 24(%rax)
	.loc 2 64 0
	movq	$0, 56(%rax)
.L837:
.LVL1681:
.LBE10279:
.LBE10284:
.LBE10268:
.LBE10267:
.LBE10266:
	.file 18 "/usr/include/c++/5/bits/stl_uninitialized.h"
	.loc 18 518 0
	addq	$96, %rax
.LVL1682:
	subq	$1, %rbx
.LVL1683:
	jne	.L838
	movq	%rsi, %rax
.LVL1684:
.L943:
.LBE10265:
.LBE10264:
.LBE10263:
.LBE10262:
.LBE10261:
.LBE10260:
.LBE10259:
.LBE10233:
	.loc 1 840 0
	movsd	8(%r15), %xmm6
.LBB10289:
.LBB10286:
.LBB10285:
	.loc 15 1310 0
	movq	%rax, -1736(%rbp)
.LVL1685:
.LBE10285:
.LBE10286:
.LBE10289:
.LBB10290:
.LBB10291:
.LBB10292:
.LBB10293:
.LBB10294:
	.loc 1 626 0
	xorl	%ebx, %ebx
.LBB10295:
	.loc 1 633 0
	movsd	.LC21(%rip), %xmm2
	leaq	-1424(%rbp), %r14
.LBB10296:
.LBB10297:
.LBB10298:
	.loc 14 750 0
	movsd	.LC22(%rip), %xmm1
.LBE10298:
.LBE10297:
.LBE10296:
.LBE10295:
.LBB10344:
	.loc 1 641 0
	xorl	%r12d, %r12d
.LBE10344:
.LBB10457:
	.loc 1 633 0
	divsd	%xmm6, %xmm2
.LBE10457:
.LBE10294:
.LBE10293:
.LBE10292:
	.loc 1 336 0
	movsd	%xmm6, -688(%rbp)
.LBB11027:
.LBB10640:
.LBB10631:
.LBB10458:
.LBB10315:
.LBB10307:
.LBB10299:
	.loc 14 750 0
	movsd	%xmm6, -1808(%rbp)
	divsd	%xmm6, %xmm1
	movapd	%xmm2, %xmm0
	movsd	%xmm2, -1752(%rbp)
.LBE10299:
.LBE10307:
.LBE10315:
.LBE10458:
.LBE10631:
.LBE10640:
.LBE11027:
.LBE10291:
.LBE10290:
	.loc 1 840 0
	movsd	16(%r15), %xmm7
.LVL1686:
	movsd	%xmm7, -1800(%rbp)
.LBB11038:
.LBB11033:
	.loc 1 336 0
	movsd	%xmm7, -680(%rbp)
.LVL1687:
.LBB11028:
.LBB10641:
.LBB10632:
.LBB10459:
.LBB10316:
.LBB10308:
.LBB10300:
	.loc 14 750 0
	call	cexp
.LVL1688:
	movsd	%xmm1, -1824(%rbp)
.LBE10300:
.LBE10308:
.LBE10316:
	.loc 1 633 0
	movsd	%xmm1, -1544(%rbp)
.LBB10317:
.LBB10318:
.LBB10319:
	.loc 14 750 0
	movsd	.LC23(%rip), %xmm1
	movsd	-1752(%rbp), %xmm2
	divsd	-1808(%rbp), %xmm1
.LBE10319:
.LBE10318:
.LBE10317:
.LBB10332:
.LBB10309:
.LBB10301:
	movsd	%xmm0, -1816(%rbp)
.LBE10301:
.LBE10309:
.LBE10332:
	.loc 1 633 0
	movsd	%xmm0, -1552(%rbp)
.LBB10333:
.LBB10326:
.LBB10320:
	.loc 14 750 0
	movapd	%xmm2, %xmm0
	call	cexp
.LVL1689:
.LBE10320:
.LBE10326:
.LBE10333:
	.loc 1 633 0
	movsd	-1808(%rbp), %xmm6
	movsd	.LC27(%rip), %xmm2
.LBB10334:
.LBB10327:
.LBB10321:
	.loc 14 750 0
	movsd	%xmm1, -1840(%rbp)
.LBE10321:
.LBE10327:
.LBE10334:
	.loc 1 633 0
	divsd	%xmm6, %xmm2
	.loc 1 634 0
	movsd	%xmm1, -1528(%rbp)
.LBB10335:
.LBB10328:
.LBB10322:
	.loc 14 750 0
	movsd	%xmm0, -1832(%rbp)
.LBE10322:
.LBE10328:
.LBE10335:
	.loc 1 634 0
	movsd	%xmm0, -1536(%rbp)
.LBB10336:
.LBB10310:
.LBB10302:
	.loc 14 750 0
	movsd	.LC28(%rip), %xmm1
.LBE10302:
.LBE10310:
.LBE10336:
	.loc 1 635 0
	movsd	.LC24(%rip), %xmm7
.LBB10337:
.LBB10311:
.LBB10303:
	.loc 14 750 0
	divsd	%xmm6, %xmm1
.LBE10303:
.LBE10311:
.LBE10337:
	.loc 1 635 0
	movsd	%xmm7, -1488(%rbp)
.LBB10338:
.LBB10312:
.LBB10304:
	.loc 14 750 0
	movapd	%xmm2, %xmm0
	movsd	%xmm2, -1752(%rbp)
.LBE10304:
.LBE10312:
.LBE10338:
	.loc 1 635 0
	movsd	.LC25(%rip), %xmm7
	.loc 1 636 0
	movsd	.LC24(%rip), %xmm3
	.loc 1 635 0
	movsd	%xmm7, -1480(%rbp)
	.loc 1 636 0
	movsd	.LC26(%rip), %xmm7
	movsd	%xmm3, -1472(%rbp)
	movsd	%xmm7, -1464(%rbp)
.LVL1690:
.LBB10339:
.LBB10313:
.LBB10305:
	.loc 14 750 0
	call	cexp
.LVL1691:
	movsd	%xmm1, -1856(%rbp)
.LBE10305:
.LBE10313:
.LBE10339:
	.loc 1 633 0
	movsd	%xmm1, -1512(%rbp)
.LBB10340:
.LBB10329:
.LBB10323:
	.loc 14 750 0
	movsd	.LC29(%rip), %xmm1
	movsd	-1752(%rbp), %xmm2
	divsd	-1808(%rbp), %xmm1
.LBE10323:
.LBE10329:
.LBE10340:
.LBB10341:
.LBB10314:
.LBB10306:
	movsd	%xmm0, -1848(%rbp)
.LBE10306:
.LBE10314:
.LBE10341:
	.loc 1 633 0
	movsd	%xmm0, -1520(%rbp)
.LBB10342:
.LBB10330:
.LBB10324:
	.loc 14 750 0
	movapd	%xmm2, %xmm0
	call	cexp
.LVL1692:
.LBE10324:
.LBE10330:
.LBE10342:
	.loc 1 635 0
	movsd	.LC30(%rip), %xmm6
.LBE10459:
	.loc 1 638 0
	leaq	-576(%rbp), %rdi
.LBB10460:
	.loc 1 635 0
	movsd	.LC31(%rip), %xmm3
.LBE10460:
	.loc 1 638 0
	movq	%rbx, %rax
.LBB10461:
	.loc 1 636 0
	movsd	.LC32(%rip), %xmm4
.LBE10461:
	.loc 1 638 0
	movl	$32, %ecx
.LBB10462:
.LBB10343:
.LBB10331:
.LBB10325:
	.loc 14 750 0
	movsd	%xmm0, -1864(%rbp)
	leaq	-576(%rbp), %rsi
	leaq	-1488(%rbp), %rbx
	movsd	%xmm1, -1872(%rbp)
.LBE10325:
.LBE10331:
.LBE10343:
	.loc 1 634 0
	movsd	%xmm0, -1504(%rbp)
	addq	$16, %rsi
	movsd	%xmm1, -1496(%rbp)
	.loc 1 635 0
	movsd	%xmm6, -1456(%rbp)
	movsd	%xmm3, -1448(%rbp)
	.loc 1 636 0
	movsd	%xmm6, -1440(%rbp)
	movsd	%xmm4, -1432(%rbp)
.LVL1693:
.LBE10462:
	.loc 1 638 0
	rep stosq
	.loc 1 639 0
	leaq	-1424(%rbp), %rdi
	movl	$8, %ecx
.LBB10463:
.LBB10345:
.LBB10346:
.LBB10347:
.LBB10348:
.LBB10349:
	.loc 14 1334 0
	movq	%r15, -1888(%rbp)
	movq	%r13, -1896(%rbp)
	movq	%rsi, %r13
.LVL1694:
.LBE10349:
.LBE10348:
.LBE10347:
.LBE10346:
.LBE10345:
.LBE10463:
	.loc 1 639 0
	rep stosq
.LVL1695:
	leaq	-1552(%rbp), %rax
.LBB10464:
.LBB10453:
.LBB10353:
.LBB10352:
.LBB10351:
.LBB10350:
	.loc 14 1334 0
	movq	%rax, %r15
.LVL1696:
.L845:
	pxor	%xmm1, %xmm1
	movsd	.LC15(%rip), %xmm0
	movsd	8(%r15), %xmm3
	movsd	(%r15), %xmm2
	call	__divdc3
.LVL1697:
.LBE10350:
.LBE10351:
.LBE10352:
.LBE10353:
.LBB10354:
.LBB10355:
	.loc 14 1254 0
	movsd	.LC33(%rip), %xmm6
.LBE10355:
.LBE10354:
	.loc 1 644 0
	testl	%r12d, %r12d
	.loc 1 642 0
	movsd	%xmm0, -16(%r13)
.LBB10358:
.LBB10356:
	.loc 14 1254 0
	movq	$0, 8(%r14)
.LBE10356:
.LBE10358:
	.loc 1 642 0
	movsd	%xmm1, -8(%r13)
.LVL1698:
.LBB10359:
.LBB10357:
	.loc 14 1254 0
	movsd	%xmm6, (%r14)
.LBE10357:
.LBE10359:
	.loc 1 644 0
	je	.L839
.LVL1699:
.LBB10360:
.LBB10361:
.LBB10362:
.LBB10363:
	.loc 14 1323 0
	movapd	%xmm1, %xmm3
	movapd	%xmm0, %xmm2
	movsd	%xmm1, -1768(%rbp)
	movsd	%xmm0, -1760(%rbp)
	movsd	-1824(%rbp), %xmm1
.LVL1700:
	movsd	-1816(%rbp), %xmm0
.LVL1701:
	call	__muldc3
.LVL1702:
.LBE10363:
.LBE10362:
.LBE10361:
.LBE10360:
.LBB10364:
.LBB10365:
	movsd	.LC15(%rip), %xmm6
	xorpd	.LC34(%rip), %xmm1
.LVL1703:
	subsd	%xmm0, %xmm6
	movsd	8(%rbx), %xmm3
	movsd	(%rbx), %xmm2
	movapd	%xmm6, %xmm0
	call	__muldc3
.LVL1704:
.LBE10365:
.LBE10364:
.LBB10369:
.LBB10370:
.LBB10371:
.LBB10372:
.LBB10373:
	movsd	-1768(%rbp), %xmm5
	movsd	-1760(%rbp), %xmm4
.LBE10373:
.LBE10372:
.LBE10371:
.LBE10370:
.LBE10369:
.LBB10444:
.LBB10366:
	movsd	%xmm1, -1752(%rbp)
.LBE10366:
.LBE10444:
.LBB10445:
.LBB10395:
.LBB10388:
.LBB10381:
.LBB10374:
	movapd	%xmm5, %xmm3
.LBE10374:
.LBE10381:
.LBE10388:
.LBE10395:
.LBE10445:
.LBB10446:
.LBB10367:
	movsd	%xmm0, (%rbx)
.LBE10367:
.LBE10446:
.LBB10447:
.LBB10396:
.LBB10389:
.LBB10382:
.LBB10375:
	movapd	%xmm4, %xmm2
	movapd	%xmm4, %xmm0
.LBE10375:
.LBE10382:
.LBE10389:
.LBE10396:
.LBE10447:
.LBB10448:
.LBB10368:
	movsd	%xmm1, 8(%rbx)
.LVL1705:
.LBE10368:
.LBE10448:
.LBB10449:
.LBB10397:
.LBB10390:
.LBB10383:
.LBB10376:
	movapd	%xmm5, %xmm1
	call	__muldc3
.LVL1706:
.LBE10376:
.LBE10383:
.LBE10390:
.LBE10397:
	.loc 1 651 0
	cmpl	$1, %r12d
	.loc 1 647 0
	movsd	%xmm0, 0(%r13)
	movsd	%xmm1, 8(%r13)
	.loc 1 651 0
	movsd	-1760(%rbp), %xmm4
	movsd	-1768(%rbp), %xmm5
	je	.L840
	movq	%r13, -1776(%rbp)
.LVL1707:
.L930:
.LBB10398:
.LBB10399:
.LBB10400:
.LBB10401:
.LBB10402:
.LBB10403:
	.loc 14 1224 0
	movsd	-16(%r13), %xmm6
.LVL1708:
.LBE10403:
.LBE10402:
.LBB10404:
.LBB10405:
	.loc 14 1228 0
	movsd	-8(%r13), %xmm7
.LVL1709:
.LBE10405:
.LBE10404:
	.loc 14 1323 0
	movapd	%xmm6, %xmm2
	movsd	-1832(%rbp), %xmm0
	movapd	%xmm7, %xmm3
	movsd	%xmm7, -1768(%rbp)
	movsd	-1840(%rbp), %xmm1
	movsd	%xmm6, -1760(%rbp)
	call	__muldc3
.LVL1710:
.LBE10401:
.LBE10400:
.LBE10399:
.LBE10398:
.LBB10415:
.LBB10416:
	movsd	.LC15(%rip), %xmm4
	xorpd	.LC34(%rip), %xmm1
.LVL1711:
	subsd	%xmm0, %xmm4
	movsd	-1752(%rbp), %xmm3
	movsd	(%rbx), %xmm2
	movapd	%xmm4, %xmm0
	call	__muldc3
.LVL1712:
.LBE10416:
.LBE10415:
.LBB10421:
.LBB10422:
.LBB10423:
.LBB10424:
	movsd	8(%r13), %xmm3
.LBE10424:
.LBE10423:
.LBE10422:
.LBE10421:
.LBB10431:
.LBB10417:
	movsd	%xmm1, -1752(%rbp)
	movsd	%xmm0, (%rbx)
	movsd	%xmm1, 8(%rbx)
.LVL1713:
.LBE10417:
.LBE10431:
.LBB10432:
.LBB10429:
.LBB10427:
.LBB10425:
	movsd	-1760(%rbp), %xmm0
	movsd	-1768(%rbp), %xmm1
	movsd	0(%r13), %xmm2
	movsd	%xmm3, -1792(%rbp)
	call	__muldc3
.LVL1714:
.LBE10425:
.LBE10427:
.LBE10429:
.LBE10432:
	.loc 1 651 0
	cmpl	$2, %r12d
	.loc 1 649 0
	movsd	%xmm0, 16(%r13)
	movsd	%xmm1, 24(%r13)
	.loc 1 651 0
	je	.L841
.L931:
.LVL1715:
.LBB10433:
.LBB10412:
.LBB10409:
.LBB10406:
	.loc 14 1323 0
	movsd	-1848(%rbp), %xmm0
	movsd	-1768(%rbp), %xmm3
	movsd	-1760(%rbp), %xmm2
	movsd	-1856(%rbp), %xmm1
	call	__muldc3
.LVL1716:
.LBE10406:
.LBE10409:
.LBE10412:
.LBE10433:
.LBB10434:
.LBB10418:
	movsd	.LC15(%rip), %xmm6
	xorpd	.LC34(%rip), %xmm1
.LVL1717:
	subsd	%xmm0, %xmm6
	movsd	-1752(%rbp), %xmm3
	movsd	(%rbx), %xmm2
	movapd	%xmm6, %xmm0
	call	__muldc3
.LVL1718:
.LBE10418:
.LBE10434:
.LBB10435:
.LBB10391:
.LBB10384:
.LBB10377:
	movq	-1776(%rbp), %rax
.LBE10377:
.LBE10384:
.LBE10391:
.LBE10435:
.LBB10436:
.LBB10419:
	movsd	%xmm1, -1752(%rbp)
	movsd	%xmm0, (%rbx)
	movsd	%xmm1, 8(%rbx)
.LVL1719:
.LBE10419:
.LBE10436:
.LBB10437:
.LBB10392:
.LBB10385:
.LBB10378:
	movsd	(%rax), %xmm0
	movsd	8(%rax), %xmm1
	movsd	-1792(%rbp), %xmm3
	movsd	0(%r13), %xmm2
	call	__muldc3
.LVL1720:
.LBE10378:
.LBE10385:
.LBE10392:
.LBE10437:
	.loc 1 651 0
	cmpl	$3, %r12d
	.loc 1 647 0
	movsd	%xmm0, 32(%r13)
	movsd	%xmm1, 40(%r13)
	.loc 1 651 0
	je	.L1009
.L842:
.LVL1721:
.LBB10438:
.LBB10413:
.LBB10410:
.LBB10407:
	.loc 14 1323 0
	movsd	-1864(%rbp), %xmm0
.LBE10407:
.LBE10410:
.LBE10413:
.LBE10438:
.LBE10449:
.LBE10453:
	.loc 1 641 0
	addl	$1, %r12d
.LVL1722:
	addq	$16, %r15
.LBB10454:
.LBB10450:
.LBB10439:
.LBB10414:
.LBB10411:
.LBB10408:
	.loc 14 1323 0
	movsd	-1768(%rbp), %xmm3
	addq	$16, %r14
	movsd	-1760(%rbp), %xmm2
	addq	$16, %rbx
	movsd	-1872(%rbp), %xmm1
	addq	$64, %r13
.LVL1723:
	call	__muldc3
.LVL1724:
.LBE10408:
.LBE10411:
.LBE10414:
.LBE10439:
.LBB10440:
.LBB10420:
	movsd	.LC15(%rip), %xmm6
	xorpd	.LC34(%rip), %xmm1
.LVL1725:
	subsd	%xmm0, %xmm6
	movsd	-16(%rbx), %xmm2
	movsd	-1752(%rbp), %xmm3
	movapd	%xmm6, %xmm0
	call	__muldc3
.LVL1726:
	movsd	%xmm0, -16(%rbx)
	movsd	%xmm1, -8(%rbx)
.LVL1727:
.LBE10420:
.LBE10440:
.LBE10450:
.LBE10454:
	.loc 1 641 0
	cmpl	$4, %r12d
	jne	.L845
.LVL1728:
.L1009:
.LBE10464:
	.loc 1 655 0
	leaq	-1584(%rbp), %rbx
	leaq	-1744(%rbp), %rax
	leaq	-576(%rbp), %rcx
	movl	$4, %r8d
	movl	$4, %edx
	movl	$4, %esi
	movq	%rbx, %r9
	movl	$101, %edi
	movq	-1888(%rbp), %r15
	movq	-1896(%rbp), %r13
.LVL1729:
	movq	%rax, -1776(%rbp)
.LEHB9:
	call	LAPACKE_zgetrf
.LVL1730:
	testl	%eax, %eax
	jne	.L1022
	.loc 1 659 0
	leaq	-1424(%rbp), %rax
.LVL1731:
	subq	$8, %rsp
	leaq	-576(%rbp), %r8
	pushq	$1
	movl	$4, %r9d
	movl	$1, %ecx
	pushq	%rax
	leaq	-1744(%rbp), %rax
	pushq	%rbx
	movl	$4, %edx
	movl	$78, %esi
	movl	$101, %edi
	movq	%rax, -1776(%rbp)
	.cfi_escape 0x2e,0x20
	call	LAPACKE_zgetrs
.LVL1732:
	addq	$32, %rsp
.LVL1733:
	testl	%eax, %eax
	jne	.L1023
.LVL1734:
.LBB10465:
.LBB10466:
.LBB10467:
	.loc 14 1254 0
	movsd	.LC15(%rip), %xmm6
	movq	$0, -568(%rbp)
.LVL1735:
.LBE10467:
.LBE10466:
.LBB10475:
.LBB10476:
.LBB10477:
.LBB10478:
	.loc 14 1334 0
	pxor	%xmm1, %xmm1
	movapd	%xmm6, %xmm0
	movsd	-1544(%rbp), %xmm3
	movsd	-1552(%rbp), %xmm2
.LBE10478:
.LBE10477:
.LBE10476:
.LBE10475:
.LBB10506:
.LBB10468:
	.loc 14 1254 0
	movsd	%xmm6, -576(%rbp)
.LBE10468:
.LBE10506:
.LBB10507:
.LBB10497:
.LBB10488:
.LBB10479:
	.loc 14 1334 0
	call	__divdc3
.LVL1736:
.LBE10479:
.LBE10488:
.LBE10497:
.LBE10507:
.LBB10508:
.LBB10509:
.LBB10510:
.LBB10511:
	.loc 14 1323 0
	movapd	%xmm1, %xmm3
.LBE10511:
.LBE10510:
.LBE10509:
.LBE10508:
	.loc 1 665 0
	movsd	%xmm0, -560(%rbp)
.LBB10545:
.LBB10534:
.LBB10523:
.LBB10512:
	.loc 14 1323 0
	movapd	%xmm0, %xmm2
.LBE10512:
.LBE10523:
.LBE10534:
.LBE10545:
	.loc 1 665 0
	movsd	%xmm1, -552(%rbp)
.LVL1737:
.LBB10546:
.LBB10535:
.LBB10524:
.LBB10513:
	.loc 14 1323 0
	movsd	%xmm1, -1760(%rbp)
	movsd	%xmm0, -1752(%rbp)
	call	__muldc3
.LVL1738:
.LBE10513:
.LBE10524:
.LBE10535:
.LBE10546:
.LBB10547:
.LBB10548:
.LBB10549:
.LBB10550:
	movsd	-1760(%rbp), %xmm5
	movsd	-1752(%rbp), %xmm4
	movapd	%xmm5, %xmm3
.LBE10550:
.LBE10549:
.LBE10548:
.LBE10547:
	.loc 1 666 0
	movsd	%xmm0, -544(%rbp)
.LBB10584:
.LBB10573:
.LBB10562:
.LBB10551:
	.loc 14 1323 0
	movapd	%xmm4, %xmm2
.LBE10551:
.LBE10562:
.LBE10573:
.LBE10584:
	.loc 1 666 0
	movsd	%xmm1, -536(%rbp)
.LVL1739:
.LBB10585:
.LBB10574:
.LBB10563:
.LBB10552:
	.loc 14 1323 0
	call	__muldc3
.LVL1740:
.LBE10552:
.LBE10563:
.LBE10574:
.LBE10585:
.LBB10586:
.LBB10469:
	.loc 14 1254 0
	movsd	.LC15(%rip), %xmm7
	movq	$0, -504(%rbp)
.LBE10469:
.LBE10586:
	.loc 1 667 0
	movsd	%xmm1, -520(%rbp)
.LVL1741:
.LBB10587:
.LBB10498:
.LBB10489:
.LBB10480:
	.loc 14 1334 0
	pxor	%xmm1, %xmm1
	movsd	-1528(%rbp), %xmm3
	movsd	-1536(%rbp), %xmm2
.LBE10480:
.LBE10489:
.LBE10498:
.LBE10587:
	.loc 1 667 0
	movsd	%xmm0, -528(%rbp)
.LBB10588:
.LBB10499:
.LBB10490:
.LBB10481:
	.loc 14 1334 0
	movapd	%xmm7, %xmm0
.LBE10481:
.LBE10490:
.LBE10499:
.LBE10588:
.LBB10589:
.LBB10470:
	.loc 14 1254 0
	movsd	%xmm7, -512(%rbp)
.LBE10470:
.LBE10589:
.LBB10590:
.LBB10500:
.LBB10491:
.LBB10482:
	.loc 14 1334 0
	call	__divdc3
.LVL1742:
.LBE10482:
.LBE10491:
.LBE10500:
.LBE10590:
.LBB10591:
.LBB10536:
.LBB10525:
.LBB10514:
	.loc 14 1323 0
	movapd	%xmm1, %xmm3
.LBE10514:
.LBE10525:
.LBE10536:
.LBE10591:
	.loc 1 665 0
	movsd	%xmm0, -496(%rbp)
.LBB10592:
.LBB10537:
.LBB10526:
.LBB10515:
	.loc 14 1323 0
	movapd	%xmm0, %xmm2
.LBE10515:
.LBE10526:
.LBE10537:
.LBE10592:
	.loc 1 665 0
	movsd	%xmm1, -488(%rbp)
.LVL1743:
.LBB10593:
.LBB10538:
.LBB10527:
.LBB10516:
	.loc 14 1323 0
	movsd	%xmm1, -1760(%rbp)
	movsd	%xmm0, -1752(%rbp)
	call	__muldc3
.LVL1744:
.LBE10516:
.LBE10527:
.LBE10538:
.LBE10593:
.LBB10594:
.LBB10575:
.LBB10564:
.LBB10553:
	movsd	-1760(%rbp), %xmm5
	movsd	-1752(%rbp), %xmm4
	movapd	%xmm5, %xmm3
.LBE10553:
.LBE10564:
.LBE10575:
.LBE10594:
	.loc 1 666 0
	movsd	%xmm0, -480(%rbp)
.LBB10595:
.LBB10576:
.LBB10565:
.LBB10554:
	.loc 14 1323 0
	movapd	%xmm4, %xmm2
.LBE10554:
.LBE10565:
.LBE10576:
.LBE10595:
	.loc 1 666 0
	movsd	%xmm1, -472(%rbp)
.LVL1745:
.LBB10596:
.LBB10577:
.LBB10566:
.LBB10555:
	.loc 14 1323 0
	call	__muldc3
.LVL1746:
.LBE10555:
.LBE10566:
.LBE10577:
.LBE10596:
.LBB10597:
.LBB10471:
	.loc 14 1254 0
	movsd	.LC15(%rip), %xmm3
	movq	$0, -440(%rbp)
.LBE10471:
.LBE10597:
	.loc 1 667 0
	movsd	%xmm1, -456(%rbp)
.LVL1747:
.LBB10598:
.LBB10501:
.LBB10492:
.LBB10483:
	.loc 14 1334 0
	pxor	%xmm1, %xmm1
	movsd	-1520(%rbp), %xmm2
.LBE10483:
.LBE10492:
.LBE10501:
.LBE10598:
	.loc 1 667 0
	movsd	%xmm0, -464(%rbp)
.LBB10599:
.LBB10472:
	.loc 14 1254 0
	movsd	%xmm3, -448(%rbp)
.LBE10472:
.LBE10599:
.LBB10600:
.LBB10502:
.LBB10493:
.LBB10484:
	.loc 14 1334 0
	movsd	.LC15(%rip), %xmm0
	movsd	-1512(%rbp), %xmm3
	call	__divdc3
.LVL1748:
.LBE10484:
.LBE10493:
.LBE10502:
.LBE10600:
.LBB10601:
.LBB10539:
.LBB10528:
.LBB10517:
	.loc 14 1323 0
	movapd	%xmm1, %xmm3
.LBE10517:
.LBE10528:
.LBE10539:
.LBE10601:
	.loc 1 665 0
	movsd	%xmm0, -432(%rbp)
.LBB10602:
.LBB10540:
.LBB10529:
.LBB10518:
	.loc 14 1323 0
	movapd	%xmm0, %xmm2
.LBE10518:
.LBE10529:
.LBE10540:
.LBE10602:
	.loc 1 665 0
	movsd	%xmm1, -424(%rbp)
.LVL1749:
.LBB10603:
.LBB10541:
.LBB10530:
.LBB10519:
	.loc 14 1323 0
	movsd	%xmm1, -1760(%rbp)
	movsd	%xmm0, -1752(%rbp)
	call	__muldc3
.LVL1750:
.LBE10519:
.LBE10530:
.LBE10541:
.LBE10603:
.LBB10604:
.LBB10578:
.LBB10567:
.LBB10556:
	movsd	-1752(%rbp), %xmm4
	movsd	-1760(%rbp), %xmm5
	movapd	%xmm4, %xmm2
.LBE10556:
.LBE10567:
.LBE10578:
.LBE10604:
	.loc 1 666 0
	movsd	%xmm0, -416(%rbp)
.LBB10605:
.LBB10579:
.LBB10568:
.LBB10557:
	.loc 14 1323 0
	movapd	%xmm5, %xmm3
.LBE10557:
.LBE10568:
.LBE10579:
.LBE10605:
	.loc 1 666 0
	movsd	%xmm1, -408(%rbp)
.LVL1751:
.LBB10606:
.LBB10580:
.LBB10569:
.LBB10558:
	.loc 14 1323 0
	call	__muldc3
.LVL1752:
.LBE10558:
.LBE10569:
.LBE10580:
.LBE10606:
.LBB10607:
.LBB10473:
	.loc 14 1254 0
	movsd	.LC15(%rip), %xmm5
	movq	$0, -376(%rbp)
.LBE10473:
.LBE10607:
	.loc 1 667 0
	movsd	%xmm1, -392(%rbp)
.LVL1753:
.LBB10608:
.LBB10503:
.LBB10494:
.LBB10485:
	.loc 14 1334 0
	pxor	%xmm1, %xmm1
.LBE10485:
.LBE10494:
.LBE10503:
.LBE10608:
	.loc 1 667 0
	movsd	%xmm0, -400(%rbp)
.LBB10609:
.LBB10504:
.LBB10495:
.LBB10486:
	.loc 14 1334 0
	movapd	%xmm5, %xmm0
	movsd	-1496(%rbp), %xmm3
	movsd	-1504(%rbp), %xmm2
.LBE10486:
.LBE10495:
.LBE10504:
.LBE10609:
.LBB10610:
.LBB10474:
	.loc 14 1254 0
	movsd	%xmm5, -384(%rbp)
.LBE10474:
.LBE10610:
.LBB10611:
.LBB10505:
.LBB10496:
.LBB10487:
	.loc 14 1334 0
	call	__divdc3
.LVL1754:
.LBE10487:
.LBE10496:
.LBE10505:
.LBE10611:
.LBB10612:
.LBB10542:
.LBB10531:
.LBB10520:
	.loc 14 1323 0
	movapd	%xmm1, %xmm3
.LBE10520:
.LBE10531:
.LBE10542:
.LBE10612:
	.loc 1 665 0
	movsd	%xmm0, -368(%rbp)
.LBB10613:
.LBB10543:
.LBB10532:
.LBB10521:
	.loc 14 1323 0
	movapd	%xmm0, %xmm2
.LBE10521:
.LBE10532:
.LBE10543:
.LBE10613:
	.loc 1 665 0
	movsd	%xmm1, -360(%rbp)
.LVL1755:
.LBB10614:
.LBB10544:
.LBB10533:
.LBB10522:
	.loc 14 1323 0
	movsd	%xmm1, -1760(%rbp)
	movsd	%xmm0, -1752(%rbp)
	call	__muldc3
.LVL1756:
.LBE10522:
.LBE10533:
.LBE10544:
.LBE10614:
.LBB10615:
.LBB10581:
.LBB10570:
.LBB10559:
	movsd	-1760(%rbp), %xmm5
	movsd	-1752(%rbp), %xmm4
	movapd	%xmm5, %xmm3
.LBE10559:
.LBE10570:
.LBE10581:
.LBE10615:
	.loc 1 666 0
	movsd	%xmm0, -352(%rbp)
.LBB10616:
.LBB10582:
.LBB10571:
.LBB10560:
	.loc 14 1323 0
	movapd	%xmm4, %xmm2
.LBE10560:
.LBE10571:
.LBE10582:
.LBE10616:
	.loc 1 666 0
	movsd	%xmm1, -344(%rbp)
.LVL1757:
.LBB10617:
.LBB10583:
.LBB10572:
.LBB10561:
	.loc 14 1323 0
	call	__muldc3
.LVL1758:
	leaq	-1744(%rbp), %rax
.LBE10561:
.LBE10572:
.LBE10583:
.LBE10617:
.LBE10465:
	.loc 1 669 0
	leaq	-576(%rbp), %rcx
	movq	%rbx, %r9
	movl	$4, %r8d
	movl	$4, %edx
	movl	$4, %esi
	movl	$101, %edi
.LBB10618:
	.loc 1 667 0
	movsd	%xmm0, -336(%rbp)
	movq	%rax, -1776(%rbp)
	movsd	%xmm1, -328(%rbp)
.LVL1759:
	.cfi_escape 0x2e,0
.LBE10618:
	.loc 1 669 0
	call	LAPACKE_zgetrf
.LVL1760:
	testl	%eax, %eax
	jne	.L1024
	.loc 1 673 0
	leaq	-1488(%rbp), %rax
.LVL1761:
	subq	$8, %rsp
	leaq	-576(%rbp), %r8
	pushq	$1
	movl	$4, %r9d
	movl	$1, %ecx
	pushq	%rax
	leaq	-1744(%rbp), %rax
	pushq	%rbx
	movl	$4, %edx
	movl	$78, %esi
	movl	$101, %edi
	movq	%rax, -1776(%rbp)
	.cfi_escape 0x2e,0x20
	call	LAPACKE_zgetrs
.LVL1762:
	addq	$32, %rsp
.LVL1763:
	testl	%eax, %eax
	jne	.L1025
	movsd	-1808(%rbp), %xmm9
	movsd	.LC38(%rip), %xmm8
.LBB10619:
	.loc 1 679 0
	movsd	-1488(%rbp), %xmm0
	mulsd	%xmm9, %xmm8
.LVL1764:
	movsd	-1456(%rbp), %xmm2
	movsd	-1440(%rbp), %xmm1
	movsd	-1472(%rbp), %xmm3
	divsd	%xmm8, %xmm0
	divsd	%xmm8, %xmm2
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -656(%rbp)
	divsd	%xmm8, %xmm1
	cvtsd2ss	%xmm2, %xmm2
	movss	%xmm2, -648(%rbp)
	.loc 1 678 0
	pxor	%xmm6, %xmm6
	.loc 1 679 0
	divsd	%xmm8, %xmm3
	.loc 1 678 0
	cvtsd2ss	-1408(%rbp), %xmm6
	movss	%xmm6, -668(%rbp)
	.loc 1 679 0
	cvtsd2ss	%xmm1, %xmm1
	movss	%xmm1, -644(%rbp)
.LBE10619:
	.loc 1 682 0
	mulss	%xmm0, %xmm6
.LBB10620:
	.loc 1 678 0
	pxor	%xmm7, %xmm7
	pxor	%xmm5, %xmm5
	pxor	%xmm4, %xmm4
	.loc 1 679 0
	cvtsd2ss	%xmm3, %xmm3
	movss	%xmm3, -652(%rbp)
	.loc 1 678 0
	cvtsd2ss	-1424(%rbp), %xmm7
	movss	%xmm7, -672(%rbp)
.LVL1765:
.LBE10620:
	.loc 1 681 0
	mulss	%xmm0, %xmm7
.LBB10621:
	.loc 1 678 0
	cvtsd2ss	-1392(%rbp), %xmm5
	movss	%xmm5, -664(%rbp)
.LVL1766:
.LBE10621:
	.loc 1 683 0
	mulss	%xmm0, %xmm5
	.loc 1 682 0
	subss	%xmm6, %xmm2
.LBE10632:
.LBE10641:
.LBB10642:
	.loc 1 695 0
	movapd	%xmm9, %xmm6
.LBE10642:
.LBB11013:
.LBB10633:
.LBB10622:
	.loc 1 678 0
	cvtsd2ss	-1376(%rbp), %xmm4
	movss	%xmm4, -660(%rbp)
.LVL1767:
.LBE10622:
	.loc 1 684 0
	xorps	.LC39(%rip), %xmm0
	.loc 1 681 0
	subss	%xmm7, %xmm3
.LBE10633:
.LBE11013:
.LBB11014:
	.loc 1 695 0
	subsd	-1800(%rbp), %xmm6
.LBE11014:
.LBB11015:
.LBB10634:
	.loc 1 683 0
	subss	%xmm5, %xmm1
	.loc 1 682 0
	movss	%xmm2, -636(%rbp)
	.loc 1 681 0
	movss	%xmm3, -640(%rbp)
	.loc 1 684 0
	mulss	%xmm4, %xmm0
	.loc 1 683 0
	movss	%xmm1, -632(%rbp)
.LBE10634:
.LBE11015:
.LBB11016:
	.loc 1 695 0
	movsd	.LC41(%rip), %xmm1
.LBE11016:
.LBB11017:
.LBB10635:
	.loc 1 684 0
	movss	%xmm0, -628(%rbp)
.LVL1768:
.LBE10635:
.LBE11017:
.LBB11018:
	.loc 1 695 0
	movapd	%xmm6, %xmm0
	andpd	%xmm1, %xmm0
	movsd	.LC42(%rip), %xmm1
	ucomisd	%xmm0, %xmm1
	ja	.L1026
.LVL1769:
.LBB10643:
.LBB10644:
.LBB10645:
	.loc 1 633 0
	movsd	-1800(%rbp), %xmm7
.LBE10645:
	.loc 1 626 0
	xorl	%ebx, %ebx
.LVL1770:
.LBB10691:
	.loc 1 633 0
	movsd	.LC21(%rip), %xmm2
.LBB10646:
.LBB10647:
.LBB10648:
	.loc 14 750 0
	movsd	.LC22(%rip), %xmm1
.LBE10648:
.LBE10647:
.LBE10646:
	.loc 1 633 0
	divsd	%xmm7, %xmm2
.LBB10663:
.LBB10656:
.LBB10649:
	.loc 14 750 0
	divsd	%xmm7, %xmm1
	movapd	%xmm2, %xmm0
	movsd	%xmm2, -1752(%rbp)
	call	cexp
.LVL1771:
	movsd	%xmm1, -1816(%rbp)
.LBE10649:
.LBE10656:
.LBE10663:
	.loc 1 633 0
	movsd	%xmm1, -1352(%rbp)
.LBB10664:
.LBB10665:
.LBB10666:
	.loc 14 750 0
	movsd	.LC23(%rip), %xmm1
	movsd	-1752(%rbp), %xmm2
	divsd	-1800(%rbp), %xmm1
.LBE10666:
.LBE10665:
.LBE10664:
.LBB10679:
.LBB10657:
.LBB10650:
	movsd	%xmm0, -1808(%rbp)
.LVL1772:
.LBE10650:
.LBE10657:
.LBE10679:
	.loc 1 633 0
	movsd	%xmm0, -1360(%rbp)
.LBB10680:
.LBB10673:
.LBB10667:
	.loc 14 750 0
	movapd	%xmm2, %xmm0
	call	cexp
.LVL1773:
.LBE10667:
.LBE10673:
.LBE10680:
	.loc 1 633 0
	movsd	-1800(%rbp), %xmm7
	movsd	.LC27(%rip), %xmm2
.LBB10681:
.LBB10674:
.LBB10668:
	.loc 14 750 0
	movsd	%xmm1, -1832(%rbp)
.LBE10668:
.LBE10674:
.LBE10681:
	.loc 1 633 0
	divsd	%xmm7, %xmm2
	.loc 1 634 0
	movsd	%xmm1, -1336(%rbp)
.LBB10682:
.LBB10675:
.LBB10669:
	.loc 14 750 0
	movsd	%xmm0, -1824(%rbp)
.LBE10669:
.LBE10675:
.LBE10682:
	.loc 1 634 0
	movsd	%xmm0, -1344(%rbp)
.LBB10683:
.LBB10658:
.LBB10651:
	.loc 14 750 0
	movsd	.LC28(%rip), %xmm1
.LBE10651:
.LBE10658:
.LBE10683:
	.loc 1 635 0
	movsd	.LC24(%rip), %xmm6
.LBB10684:
.LBB10659:
.LBB10652:
	.loc 14 750 0
	divsd	%xmm7, %xmm1
	movapd	%xmm2, %xmm0
.LBE10652:
.LBE10659:
.LBE10684:
	.loc 1 635 0
	movsd	%xmm6, -1296(%rbp)
	.loc 1 636 0
	movsd	%xmm6, -1280(%rbp)
.LBB10685:
.LBB10660:
.LBB10653:
	.loc 14 750 0
	movsd	%xmm2, -1752(%rbp)
.LBE10653:
.LBE10660:
.LBE10685:
	.loc 1 635 0
	movsd	.LC25(%rip), %xmm3
	.loc 1 636 0
	movsd	.LC26(%rip), %xmm4
	.loc 1 635 0
	movsd	%xmm3, -1288(%rbp)
	.loc 1 636 0
	movsd	%xmm4, -1272(%rbp)
.LVL1774:
.LBB10686:
.LBB10661:
.LBB10654:
	.loc 14 750 0
	call	cexp
.LVL1775:
	movsd	%xmm1, -1848(%rbp)
.LBE10654:
.LBE10661:
.LBE10686:
	.loc 1 633 0
	movsd	%xmm1, -1320(%rbp)
.LBB10687:
.LBB10676:
.LBB10670:
	.loc 14 750 0
	movsd	.LC29(%rip), %xmm1
	movsd	-1752(%rbp), %xmm2
	divsd	-1800(%rbp), %xmm1
.LBE10670:
.LBE10676:
.LBE10687:
.LBB10688:
.LBB10662:
.LBB10655:
	movsd	%xmm0, -1840(%rbp)
.LBE10655:
.LBE10662:
.LBE10688:
	.loc 1 633 0
	movsd	%xmm0, -1328(%rbp)
.LBB10689:
.LBB10677:
.LBB10671:
	.loc 14 750 0
	movapd	%xmm2, %xmm0
	call	cexp
.LVL1776:
.LBE10671:
.LBE10677:
.LBE10689:
.LBE10691:
	.loc 1 638 0
	leaq	-320(%rbp), %rdi
	movq	%rbx, %rax
	movl	$32, %ecx
.LBB10692:
	.loc 1 635 0
	movsd	.LC30(%rip), %xmm7
	leaq	-1232(%rbp), %rsi
.LBE10692:
	.loc 1 638 0
	rep stosq
	.loc 1 639 0
	leaq	-1232(%rbp), %rdi
	movl	$8, %ecx
	leaq	-1296(%rbp), %rbx
.LBB10693:
	.loc 1 635 0
	movsd	.LC31(%rip), %xmm6
.LBE10693:
.LBB10694:
.LBB10695:
.LBB10696:
.LBB10697:
.LBB10698:
.LBB10699:
	.loc 14 1334 0
	movq	%rsi, %r12
.LBE10699:
.LBE10698:
.LBE10697:
.LBE10696:
.LBE10695:
.LBE10694:
.LBB10821:
	.loc 1 636 0
	movsd	.LC32(%rip), %xmm3
.LBB10690:
.LBB10678:
.LBB10672:
	.loc 14 750 0
	movsd	%xmm0, -1856(%rbp)
	movsd	%xmm1, -1864(%rbp)
.LBE10672:
.LBE10678:
.LBE10690:
.LBE10821:
	.loc 1 639 0
	rep stosq
	leaq	-320(%rbp), %rdi
	leaq	-1360(%rbp), %rax
.LBB10822:
	.loc 1 634 0
	movsd	%xmm0, -1312(%rbp)
	movsd	%xmm1, -1304(%rbp)
	leaq	16(%rdi), %r14
.LBE10822:
.LBB10823:
	.loc 1 641 0
	xorl	%edi, %edi
.LBE10823:
.LBB10824:
	.loc 1 635 0
	movsd	%xmm7, -1264(%rbp)
	movsd	%xmm6, -1256(%rbp)
	.loc 1 636 0
	movsd	%xmm7, -1248(%rbp)
	movsd	%xmm3, -1240(%rbp)
.LVL1777:
.LBE10824:
.LBB10825:
.LBB10817:
.LBB10703:
.LBB10702:
.LBB10701:
.LBB10700:
	.loc 14 1334 0
	movl	%edi, -1760(%rbp)
	movq	%r13, -1888(%rbp)
.LVL1778:
	movq	%rax, %r13
	movq	%r15, -1872(%rbp)
.LVL1779:
.L860:
	pxor	%xmm1, %xmm1
	movsd	8(%r13), %xmm3
	movsd	.LC15(%rip), %xmm0
	movsd	0(%r13), %xmm2
	call	__divdc3
.LVL1780:
.LBE10700:
.LBE10701:
.LBE10702:
.LBE10703:
	.loc 1 644 0
	movl	-1760(%rbp), %r15d
.LBB10704:
.LBB10705:
	.loc 14 1254 0
	movsd	.LC33(%rip), %xmm7
.LBE10705:
.LBE10704:
	.loc 1 642 0
	movsd	%xmm0, -16(%r14)
.LBB10708:
.LBB10706:
	.loc 14 1254 0
	movq	$0, 8(%r12)
.LBE10706:
.LBE10708:
	.loc 1 642 0
	movsd	%xmm1, -8(%r14)
.LVL1781:
.LBB10709:
.LBB10710:
.LBB10711:
.LBB10712:
	.loc 14 1323 0
	movapd	%xmm1, %xmm3
.LBE10712:
.LBE10711:
.LBE10710:
.LBE10709:
	.loc 1 644 0
	testl	%r15d, %r15d
.LBB10716:
.LBB10707:
	.loc 14 1254 0
	movsd	%xmm7, (%r12)
.LBE10707:
.LBE10716:
	.loc 1 644 0
	je	.L854
.LVL1782:
.LBB10717:
.LBB10715:
.LBB10714:
.LBB10713:
	.loc 14 1323 0
	movapd	%xmm0, %xmm2
	movsd	%xmm1, -1776(%rbp)
	movsd	%xmm0, -1768(%rbp)
	movsd	-1816(%rbp), %xmm1
.LVL1783:
	movsd	-1808(%rbp), %xmm0
.LVL1784:
	call	__muldc3
.LVL1785:
.LBE10713:
.LBE10714:
.LBE10715:
.LBE10717:
.LBB10718:
.LBB10719:
	movsd	.LC15(%rip), %xmm7
	xorpd	.LC34(%rip), %xmm1
.LVL1786:
	subsd	%xmm0, %xmm7
	movsd	8(%rbx), %xmm3
	movsd	(%rbx), %xmm2
	movapd	%xmm7, %xmm0
	call	__muldc3
.LVL1787:
.LBE10719:
.LBE10718:
.LBB10723:
.LBB10724:
.LBB10725:
.LBB10726:
.LBB10727:
	movsd	-1776(%rbp), %xmm5
	movsd	-1768(%rbp), %xmm4
.LBE10727:
.LBE10726:
.LBE10725:
.LBE10724:
.LBE10723:
.LBB10808:
.LBB10720:
	movsd	%xmm1, -1752(%rbp)
.LBE10720:
.LBE10808:
.LBB10809:
.LBB10749:
.LBB10742:
.LBB10735:
.LBB10728:
	movapd	%xmm5, %xmm3
.LBE10728:
.LBE10735:
.LBE10742:
.LBE10749:
.LBE10809:
.LBB10810:
.LBB10721:
	movsd	%xmm0, (%rbx)
.LBE10721:
.LBE10810:
.LBB10811:
.LBB10750:
.LBB10743:
.LBB10736:
.LBB10729:
	movapd	%xmm4, %xmm2
	movapd	%xmm4, %xmm0
.LBE10729:
.LBE10736:
.LBE10743:
.LBE10750:
.LBE10811:
.LBB10812:
.LBB10722:
	movsd	%xmm1, 8(%rbx)
.LVL1788:
.LBE10722:
.LBE10812:
.LBB10813:
.LBB10751:
.LBB10744:
.LBB10737:
.LBB10730:
	movapd	%xmm5, %xmm1
	call	__muldc3
.LVL1789:
.LBE10730:
.LBE10737:
.LBE10744:
.LBE10751:
	.loc 1 651 0
	cmpl	$1, %r15d
	.loc 1 647 0
	movsd	%xmm0, (%r14)
	movsd	%xmm1, 8(%r14)
	.loc 1 651 0
	movsd	-1768(%rbp), %xmm4
	movsd	-1776(%rbp), %xmm5
	je	.L855
	movq	%r14, %r15
.LVL1790:
.L932:
.LBB10752:
.LBB10753:
.LBB10754:
.LBB10755:
.LBB10756:
.LBB10757:
	.loc 14 1224 0
	movsd	-16(%r14), %xmm6
.LVL1791:
.LBE10757:
.LBE10756:
.LBB10758:
.LBB10759:
	.loc 14 1228 0
	movsd	-8(%r14), %xmm7
.LVL1792:
.LBE10759:
.LBE10758:
	.loc 14 1323 0
	movapd	%xmm6, %xmm0
	movsd	-1832(%rbp), %xmm3
	movapd	%xmm7, %xmm1
	movsd	%xmm7, -1776(%rbp)
	movsd	-1824(%rbp), %xmm2
	movsd	%xmm6, -1768(%rbp)
	call	__muldc3
.LVL1793:
.LBE10755:
.LBE10754:
.LBE10753:
.LBE10752:
.LBB10766:
.LBB10767:
	movsd	.LC15(%rip), %xmm4
	xorpd	.LC34(%rip), %xmm1
.LVL1794:
	subsd	%xmm0, %xmm4
	movsd	-1752(%rbp), %xmm3
	movsd	(%rbx), %xmm2
	movapd	%xmm4, %xmm0
	call	__muldc3
.LVL1795:
.LBE10767:
.LBE10766:
.LBB10775:
.LBB10776:
.LBB10777:
.LBB10778:
	movsd	8(%r14), %xmm5
.LBE10778:
.LBE10777:
.LBE10776:
.LBE10775:
.LBB10791:
.LBB10768:
	movsd	%xmm1, -1752(%rbp)
	movsd	%xmm0, (%rbx)
.LBE10768:
.LBE10791:
.LBB10792:
.LBB10787:
.LBB10783:
.LBB10779:
	movapd	%xmm5, %xmm3
.LBE10779:
.LBE10783:
.LBE10787:
.LBE10792:
.LBB10793:
.LBB10769:
	movsd	%xmm1, 8(%rbx)
.LVL1796:
.LBE10769:
.LBE10793:
.LBB10794:
.LBB10788:
.LBB10784:
.LBB10780:
	movsd	-1768(%rbp), %xmm0
	movsd	-1776(%rbp), %xmm1
	movsd	(%r14), %xmm2
	movsd	%xmm5, -1792(%rbp)
	call	__muldc3
.LVL1797:
.LBE10780:
.LBE10784:
.LBE10788:
.LBE10794:
	.loc 1 651 0
	cmpl	$2, -1760(%rbp)
	.loc 1 649 0
	movsd	%xmm0, 16(%r14)
	movsd	%xmm1, 24(%r14)
	.loc 1 651 0
	je	.L856
.L933:
.LVL1798:
.LBB10795:
.LBB10764:
.LBB10762:
.LBB10760:
	.loc 14 1323 0
	movsd	-1840(%rbp), %xmm0
	movsd	-1776(%rbp), %xmm3
	movsd	-1768(%rbp), %xmm2
	movsd	-1848(%rbp), %xmm1
	call	__muldc3
.LVL1799:
.LBE10760:
.LBE10762:
.LBE10764:
.LBE10795:
.LBB10796:
.LBB10770:
	movsd	.LC15(%rip), %xmm7
	xorpd	.LC34(%rip), %xmm1
.LVL1800:
	subsd	%xmm0, %xmm7
	movsd	-1752(%rbp), %xmm3
	movsd	(%rbx), %xmm2
	movapd	%xmm7, %xmm0
	call	__muldc3
.LVL1801:
.LBE10770:
.LBE10796:
.LBB10797:
.LBB10745:
.LBB10738:
.LBB10731:
	movsd	-1792(%rbp), %xmm3
.LBE10731:
.LBE10738:
.LBE10745:
.LBE10797:
.LBB10798:
.LBB10771:
	movsd	%xmm1, -1752(%rbp)
	movsd	%xmm0, (%rbx)
	movsd	%xmm1, 8(%rbx)
.LVL1802:
.LBE10771:
.LBE10798:
.LBB10799:
.LBB10746:
.LBB10739:
.LBB10732:
	movsd	(%r15), %xmm0
	movsd	8(%r15), %xmm1
	movsd	(%r14), %xmm2
	call	__muldc3
.LVL1803:
.LBE10732:
.LBE10739:
.LBE10746:
.LBE10799:
	.loc 1 651 0
	cmpl	$3, -1760(%rbp)
	.loc 1 647 0
	movsd	%xmm0, 32(%r14)
	movsd	%xmm1, 40(%r14)
	.loc 1 651 0
	je	.L1016
.L857:
.LVL1804:
.LBB10800:
.LBB10765:
.LBB10763:
.LBB10761:
	.loc 14 1323 0
	movsd	-1856(%rbp), %xmm0
	addq	$16, %r13
	addq	$16, %r12
	movsd	-1776(%rbp), %xmm3
	addq	$16, %rbx
	movsd	-1768(%rbp), %xmm2
	addq	$64, %r14
.LVL1805:
	movsd	-1864(%rbp), %xmm1
	call	__muldc3
.LVL1806:
.LBE10761:
.LBE10763:
.LBE10765:
.LBE10800:
.LBB10801:
.LBB10772:
	movsd	.LC15(%rip), %xmm6
	xorpd	.LC34(%rip), %xmm1
.LVL1807:
	subsd	%xmm0, %xmm6
	movsd	-16(%rbx), %xmm2
	movsd	-1752(%rbp), %xmm3
	movapd	%xmm6, %xmm0
	call	__muldc3
.LVL1808:
.LBE10772:
.LBE10801:
.LBE10813:
.LBE10817:
	.loc 1 641 0
	addl	$1, -1760(%rbp)
.LVL1809:
.LBB10818:
.LBB10814:
.LBB10802:
.LBB10773:
	.loc 14 1323 0
	movsd	%xmm0, -16(%rbx)
.LBE10773:
.LBE10802:
.LBE10814:
.LBE10818:
	.loc 1 641 0
	movl	-1760(%rbp), %eax
.LBB10819:
.LBB10815:
.LBB10803:
.LBB10774:
	.loc 14 1323 0
	movsd	%xmm1, -8(%rbx)
.LVL1810:
.LBE10774:
.LBE10803:
.LBE10815:
.LBE10819:
	.loc 1 641 0
	cmpl	$4, %eax
	jne	.L860
.LVL1811:
.L1016:
.LBE10825:
	.loc 1 655 0
	leaq	-1568(%rbp), %rbx
	leaq	-1744(%rbp), %rax
	leaq	-320(%rbp), %rcx
	movl	$4, %r8d
	movl	$4, %edx
	movl	$4, %esi
	movq	%rbx, %r9
	movl	$101, %edi
	movq	-1872(%rbp), %r15
	movq	-1888(%rbp), %r13
.LVL1812:
	movq	%rax, -1776(%rbp)
	.cfi_escape 0x2e,0
	call	LAPACKE_zgetrf
.LVL1813:
	testl	%eax, %eax
	jne	.L1027
	.loc 1 659 0
	leaq	-1232(%rbp), %rax
.LVL1814:
	subq	$8, %rsp
	leaq	-320(%rbp), %r8
	pushq	$1
	movl	$4, %r9d
	movl	$1, %ecx
	pushq	%rax
	leaq	-1744(%rbp), %rax
	pushq	%rbx
	movl	$4, %edx
	movl	$78, %esi
	movl	$101, %edi
	movq	%rax, -1776(%rbp)
	.cfi_escape 0x2e,0x20
	call	LAPACKE_zgetrs
.LVL1815:
	addq	$32, %rsp
.LVL1816:
	testl	%eax, %eax
	jne	.L1028
.LVL1817:
.LBB10826:
.LBB10827:
.LBB10828:
	.loc 14 1254 0
	movsd	.LC15(%rip), %xmm5
	movq	$0, -312(%rbp)
.LVL1818:
.LBE10828:
.LBE10827:
.LBB10836:
.LBB10837:
.LBB10838:
.LBB10839:
	.loc 14 1334 0
	pxor	%xmm1, %xmm1
	movapd	%xmm5, %xmm0
	movsd	-1352(%rbp), %xmm3
	movsd	-1360(%rbp), %xmm2
.LBE10839:
.LBE10838:
.LBE10837:
.LBE10836:
.LBB10870:
.LBB10829:
	.loc 14 1254 0
	movsd	%xmm5, -320(%rbp)
.LBE10829:
.LBE10870:
.LBB10871:
.LBB10860:
.LBB10850:
.LBB10840:
	.loc 14 1334 0
	call	__divdc3
.LVL1819:
.LBE10840:
.LBE10850:
.LBE10860:
.LBE10871:
.LBB10872:
.LBB10873:
.LBB10874:
.LBB10875:
	.loc 14 1323 0
	movapd	%xmm1, %xmm3
.LBE10875:
.LBE10874:
.LBE10873:
.LBE10872:
	.loc 1 665 0
	movsd	%xmm0, -304(%rbp)
.LBB10909:
.LBB10898:
.LBB10887:
.LBB10876:
	.loc 14 1323 0
	movapd	%xmm0, %xmm2
.LBE10876:
.LBE10887:
.LBE10898:
.LBE10909:
	.loc 1 665 0
	movsd	%xmm1, -296(%rbp)
.LVL1820:
.LBB10910:
.LBB10899:
.LBB10888:
.LBB10877:
	.loc 14 1323 0
	movsd	%xmm1, -1760(%rbp)
	movsd	%xmm0, -1752(%rbp)
	call	__muldc3
.LVL1821:
.LBE10877:
.LBE10888:
.LBE10899:
.LBE10910:
.LBB10911:
.LBB10912:
.LBB10913:
.LBB10914:
	movsd	-1760(%rbp), %xmm5
	movsd	-1752(%rbp), %xmm4
	movapd	%xmm5, %xmm3
.LBE10914:
.LBE10913:
.LBE10912:
.LBE10911:
	.loc 1 666 0
	movsd	%xmm0, -288(%rbp)
.LBB10948:
.LBB10937:
.LBB10926:
.LBB10915:
	.loc 14 1323 0
	movapd	%xmm4, %xmm2
.LBE10915:
.LBE10926:
.LBE10937:
.LBE10948:
	.loc 1 666 0
	movsd	%xmm1, -280(%rbp)
.LVL1822:
.LBB10949:
.LBB10938:
.LBB10927:
.LBB10916:
	.loc 14 1323 0
	call	__muldc3
.LVL1823:
.LBE10916:
.LBE10927:
.LBE10938:
.LBE10949:
.LBB10950:
.LBB10830:
	.loc 14 1254 0
	movsd	.LC15(%rip), %xmm4
	movq	$0, -248(%rbp)
.LBE10830:
.LBE10950:
	.loc 1 667 0
	movsd	%xmm1, -264(%rbp)
.LVL1824:
.LBB10951:
.LBB10861:
.LBB10851:
.LBB10841:
	.loc 14 1334 0
	pxor	%xmm1, %xmm1
	movsd	-1336(%rbp), %xmm3
	movsd	-1344(%rbp), %xmm2
.LBE10841:
.LBE10851:
.LBE10861:
.LBE10951:
	.loc 1 667 0
	movsd	%xmm0, -272(%rbp)
.LBB10952:
.LBB10862:
.LBB10852:
.LBB10842:
	.loc 14 1334 0
	movapd	%xmm4, %xmm0
.LBE10842:
.LBE10852:
.LBE10862:
.LBE10952:
.LBB10953:
.LBB10831:
	.loc 14 1254 0
	movsd	%xmm4, -256(%rbp)
.LBE10831:
.LBE10953:
.LBB10954:
.LBB10863:
.LBB10853:
.LBB10843:
	.loc 14 1334 0
	call	__divdc3
.LVL1825:
.LBE10843:
.LBE10853:
.LBE10863:
.LBE10954:
.LBB10955:
.LBB10900:
.LBB10889:
.LBB10878:
	.loc 14 1323 0
	movapd	%xmm1, %xmm3
.LBE10878:
.LBE10889:
.LBE10900:
.LBE10955:
	.loc 1 665 0
	movsd	%xmm0, -240(%rbp)
.LBB10956:
.LBB10901:
.LBB10890:
.LBB10879:
	.loc 14 1323 0
	movapd	%xmm0, %xmm2
.LBE10879:
.LBE10890:
.LBE10901:
.LBE10956:
	.loc 1 665 0
	movsd	%xmm1, -232(%rbp)
.LVL1826:
.LBB10957:
.LBB10902:
.LBB10891:
.LBB10880:
	.loc 14 1323 0
	movsd	%xmm1, -1760(%rbp)
	movsd	%xmm0, -1752(%rbp)
	call	__muldc3
.LVL1827:
.LBE10880:
.LBE10891:
.LBE10902:
.LBE10957:
.LBB10958:
.LBB10939:
.LBB10928:
.LBB10917:
	movsd	-1760(%rbp), %xmm5
	movsd	-1752(%rbp), %xmm4
	movapd	%xmm5, %xmm3
.LBE10917:
.LBE10928:
.LBE10939:
.LBE10958:
	.loc 1 666 0
	movsd	%xmm0, -224(%rbp)
.LBB10959:
.LBB10940:
.LBB10929:
.LBB10918:
	.loc 14 1323 0
	movapd	%xmm4, %xmm2
.LBE10918:
.LBE10929:
.LBE10940:
.LBE10959:
	.loc 1 666 0
	movsd	%xmm1, -216(%rbp)
.LVL1828:
.LBB10960:
.LBB10941:
.LBB10930:
.LBB10919:
	.loc 14 1323 0
	call	__muldc3
.LVL1829:
.LBE10919:
.LBE10930:
.LBE10941:
.LBE10960:
.LBB10961:
.LBB10832:
	.loc 14 1254 0
	movsd	.LC15(%rip), %xmm6
	movq	$0, -184(%rbp)
.LBE10832:
.LBE10961:
	.loc 1 667 0
	movsd	%xmm1, -200(%rbp)
.LVL1830:
.LBB10962:
.LBB10864:
.LBB10854:
.LBB10844:
	.loc 14 1334 0
	pxor	%xmm1, %xmm1
	movsd	-1320(%rbp), %xmm3
	movsd	-1328(%rbp), %xmm2
.LBE10844:
.LBE10854:
.LBE10864:
.LBE10962:
	.loc 1 667 0
	movsd	%xmm0, -208(%rbp)
.LBB10963:
.LBB10865:
.LBB10855:
.LBB10845:
	.loc 14 1334 0
	movapd	%xmm6, %xmm0
.LBE10845:
.LBE10855:
.LBE10865:
.LBE10963:
.LBB10964:
.LBB10833:
	.loc 14 1254 0
	movsd	%xmm6, -192(%rbp)
.LBE10833:
.LBE10964:
.LBB10965:
.LBB10866:
.LBB10856:
.LBB10846:
	.loc 14 1334 0
	call	__divdc3
.LVL1831:
.LBE10846:
.LBE10856:
.LBE10866:
.LBE10965:
.LBB10966:
.LBB10903:
.LBB10892:
.LBB10881:
	.loc 14 1323 0
	movapd	%xmm1, %xmm3
.LBE10881:
.LBE10892:
.LBE10903:
.LBE10966:
	.loc 1 665 0
	movsd	%xmm0, -176(%rbp)
.LBB10967:
.LBB10904:
.LBB10893:
.LBB10882:
	.loc 14 1323 0
	movapd	%xmm0, %xmm2
.LBE10882:
.LBE10893:
.LBE10904:
.LBE10967:
	.loc 1 665 0
	movsd	%xmm1, -168(%rbp)
.LVL1832:
.LBB10968:
.LBB10905:
.LBB10894:
.LBB10883:
	.loc 14 1323 0
	movsd	%xmm1, -1760(%rbp)
	movsd	%xmm0, -1752(%rbp)
	call	__muldc3
.LVL1833:
.LBE10883:
.LBE10894:
.LBE10905:
.LBE10968:
.LBB10969:
.LBB10942:
.LBB10931:
.LBB10920:
	movsd	-1760(%rbp), %xmm5
	movsd	-1752(%rbp), %xmm4
	movapd	%xmm5, %xmm3
.LBE10920:
.LBE10931:
.LBE10942:
.LBE10969:
	.loc 1 666 0
	movsd	%xmm0, -160(%rbp)
.LBB10970:
.LBB10943:
.LBB10932:
.LBB10921:
	.loc 14 1323 0
	movapd	%xmm4, %xmm2
.LBE10921:
.LBE10932:
.LBE10943:
.LBE10970:
	.loc 1 666 0
	movsd	%xmm1, -152(%rbp)
.LVL1834:
.LBB10971:
.LBB10944:
.LBB10933:
.LBB10922:
	.loc 14 1323 0
	call	__muldc3
.LVL1835:
.LBE10922:
.LBE10933:
.LBE10944:
.LBE10971:
.LBB10972:
.LBB10834:
	.loc 14 1254 0
	movsd	.LC15(%rip), %xmm7
	movq	$0, -120(%rbp)
.LBE10834:
.LBE10972:
	.loc 1 667 0
	movsd	%xmm1, -136(%rbp)
.LVL1836:
.LBB10973:
.LBB10867:
.LBB10857:
.LBB10847:
	.loc 14 1334 0
	pxor	%xmm1, %xmm1
	movsd	-1304(%rbp), %xmm3
	movsd	-1312(%rbp), %xmm2
.LBE10847:
.LBE10857:
.LBE10867:
.LBE10973:
	.loc 1 667 0
	movsd	%xmm0, -144(%rbp)
.LBB10974:
.LBB10868:
.LBB10858:
.LBB10848:
	.loc 14 1334 0
	movapd	%xmm7, %xmm0
.LBE10848:
.LBE10858:
.LBE10868:
.LBE10974:
.LBB10975:
.LBB10835:
	.loc 14 1254 0
	movsd	%xmm7, -128(%rbp)
.LBE10835:
.LBE10975:
.LBB10976:
.LBB10869:
.LBB10859:
.LBB10849:
	.loc 14 1334 0
	call	__divdc3
.LVL1837:
.LBE10849:
.LBE10859:
.LBE10869:
.LBE10976:
.LBB10977:
.LBB10906:
.LBB10895:
.LBB10884:
	.loc 14 1323 0
	movapd	%xmm1, %xmm3
.LBE10884:
.LBE10895:
.LBE10906:
.LBE10977:
	.loc 1 665 0
	movsd	%xmm0, -112(%rbp)
.LBB10978:
.LBB10907:
.LBB10896:
.LBB10885:
	.loc 14 1323 0
	movapd	%xmm0, %xmm2
.LBE10885:
.LBE10896:
.LBE10907:
.LBE10978:
	.loc 1 665 0
	movsd	%xmm1, -104(%rbp)
.LVL1838:
.LBB10979:
.LBB10908:
.LBB10897:
.LBB10886:
	.loc 14 1323 0
	movsd	%xmm1, -1760(%rbp)
	movsd	%xmm0, -1752(%rbp)
	call	__muldc3
.LVL1839:
.LBE10886:
.LBE10897:
.LBE10908:
.LBE10979:
.LBB10980:
.LBB10945:
.LBB10934:
.LBB10923:
	movsd	-1760(%rbp), %xmm5
	movsd	-1752(%rbp), %xmm4
	movapd	%xmm5, %xmm3
.LBE10923:
.LBE10934:
.LBE10945:
.LBE10980:
	.loc 1 666 0
	movsd	%xmm0, -96(%rbp)
.LBB10981:
.LBB10946:
.LBB10935:
.LBB10924:
	.loc 14 1323 0
	movapd	%xmm4, %xmm2
.LBE10924:
.LBE10935:
.LBE10946:
.LBE10981:
	.loc 1 666 0
	movsd	%xmm1, -88(%rbp)
.LVL1840:
.LBB10982:
.LBB10947:
.LBB10936:
.LBB10925:
	.loc 14 1323 0
	call	__muldc3
.LVL1841:
	leaq	-1744(%rbp), %rax
.LBE10925:
.LBE10936:
.LBE10947:
.LBE10982:
.LBE10826:
	.loc 1 669 0
	leaq	-320(%rbp), %rcx
	movq	%rbx, %r9
	movl	$4, %r8d
	movl	$4, %edx
	movl	$4, %esi
	movl	$101, %edi
.LBB10983:
	.loc 1 667 0
	movsd	%xmm0, -80(%rbp)
	movq	%rax, -1776(%rbp)
	movsd	%xmm1, -72(%rbp)
.LVL1842:
	.cfi_escape 0x2e,0
.LBE10983:
	.loc 1 669 0
	call	LAPACKE_zgetrf
.LVL1843:
	testl	%eax, %eax
	jne	.L1029
	.loc 1 673 0
	leaq	-1296(%rbp), %rax
.LVL1844:
	subq	$8, %rsp
	leaq	-320(%rbp), %r8
	pushq	$1
	movl	$4, %r9d
	movl	$1, %ecx
	pushq	%rax
	leaq	-1744(%rbp), %rax
	pushq	%rbx
	movl	$4, %edx
	movl	$78, %esi
	movl	$101, %edi
	movq	%rax, -1776(%rbp)
	.cfi_escape 0x2e,0x20
	call	LAPACKE_zgetrs
.LVL1845:
	addq	$32, %rsp
.LVL1846:
	testl	%eax, %eax
	jne	.L1030
	movsd	.LC38(%rip), %xmm4
.LBB10984:
	.loc 1 679 0
	movsd	-1296(%rbp), %xmm0
	mulsd	-1800(%rbp), %xmm4
.LVL1847:
	movsd	-1280(%rbp), %xmm3
	movsd	-1264(%rbp), %xmm2
	movsd	-1248(%rbp), %xmm1
	.loc 1 678 0
	pxor	%xmm8, %xmm8
	pxor	%xmm7, %xmm7
	.loc 1 679 0
	divsd	%xmm4, %xmm0
	.loc 1 678 0
	cvtsd2ss	-1232(%rbp), %xmm8
	movss	%xmm8, -624(%rbp)
	cvtsd2ss	-1216(%rbp), %xmm7
	movss	%xmm7, -620(%rbp)
	.loc 1 679 0
	divsd	%xmm4, %xmm3
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -608(%rbp)
.LVL1848:
	divsd	%xmm4, %xmm2
	cvtsd2ss	%xmm3, %xmm3
	movss	%xmm3, -604(%rbp)
.LVL1849:
	divsd	%xmm4, %xmm1
	cvtsd2ss	%xmm2, %xmm2
	movss	%xmm2, -600(%rbp)
	.loc 1 678 0
	pxor	%xmm6, %xmm6
.LBE10984:
	.loc 1 681 0
	mulss	%xmm0, %xmm8
	.loc 1 682 0
	mulss	%xmm0, %xmm7
.LBB10985:
	.loc 1 678 0
	cvtsd2ss	-1200(%rbp), %xmm6
	movss	%xmm6, -616(%rbp)
.LVL1850:
.LBE10985:
	.loc 1 683 0
	mulss	%xmm0, %xmm6
	.loc 1 684 0
	xorps	.LC39(%rip), %xmm0
.LBB10986:
	.loc 1 679 0
	cvtsd2ss	%xmm1, %xmm1
	movss	%xmm1, -596(%rbp)
.LBE10986:
	.loc 1 681 0
	subss	%xmm8, %xmm3
.LBB10987:
	.loc 1 678 0
	pxor	%xmm5, %xmm5
.LBE10987:
	.loc 1 682 0
	subss	%xmm7, %xmm2
	.loc 1 683 0
	subss	%xmm6, %xmm1
.LBB10988:
	.loc 1 678 0
	cvtsd2ss	-1184(%rbp), %xmm5
.LBE10988:
	.loc 1 684 0
	mulss	%xmm5, %xmm0
.LBB10989:
	.loc 1 678 0
	movss	%xmm5, -612(%rbp)
.LVL1851:
.LBE10989:
	.loc 1 681 0
	movss	%xmm3, -592(%rbp)
	.loc 1 682 0
	movss	%xmm2, -588(%rbp)
	.loc 1 683 0
	movss	%xmm1, -584(%rbp)
	.loc 1 684 0
	movss	%xmm0, -580(%rbp)
.LVL1852:
.L853:
.LBE10644:
.LBE10643:
.LBE11018:
.LBE11028:
.LBE11033:
.LBE11038:
	.loc 1 842 0
	movl	0(%r13), %ecx
.LBB11039:
	.loc 1 846 0
	movl	-1780(%rbp), %r10d
.LBE11039:
	.loc 1 841 0
	movl	8(%r13), %eax
	.loc 1 842 0
	movl	%ecx, %r14d
	andl	$4088, %r14d
	sarl	$3, %r14d
	.loc 1 841 0
	movl	%eax, -1760(%rbp)
.LVL1853:
	.loc 1 842 0
	addl	$1, %r14d
	imull	12(%r13), %r14d
.LVL1854:
.LBB11117:
	.loc 1 846 0
	testl	%r10d, %r10d
	jle	.L1031
	leaq	-1744(%rbp), %rax
.LVL1855:
	movl	$0, -1752(%rbp)
	movq	%rax, -1776(%rbp)
.LBB11040:
.LBB11041:
.LBB11042:
.LBB11043:
	.loc 2 709 0
	leaq	-784(%rbp), %rax
	addq	$8, %rax
	movq	%rax, -1792(%rbp)
.LBE11043:
.LBE11042:
.LBB11046:
.LBB11047:
	.loc 2 738 0
	leaq	-784(%rbp), %rax
	addq	$80, %rax
	movq	%rax, -1768(%rbp)
.LVL1856:
	.p2align 4,,10
	.p2align 3
.L882:
.LBE11047:
.LBE11046:
.LBE11041:
.LBB11068:
.LBB11069:
.LBB11070:
.LBB11071:
.LBB11072:
.LBB11073:
	.loc 3 63 0
	pxor	%xmm0, %xmm0
	cvtsi2sd	-1752(%rbp), %xmm0
	mulsd	(%r15), %xmm0
.LBE11073:
.LBE11072:
.LBB11074:
.LBB11075:
	.loc 3 827 0
	cvtsd2si	%xmm0, %eax
.LVL1857:
.LBE11075:
.LBE11074:
.LBE11071:
.LBE11070:
.LBB11076:
.LBB11077:
	.loc 13 132 0
	cmpl	$255, %eax
	movl	%eax, %r12d
	jbe	.L867
	testl	%eax, %eax
	setg	%r12b
	negl	%r12d
.L867:
.LVL1858:
.LBE11077:
.LBE11076:
.LBE11069:
.LBE11068:
.LBB11078:
.LBB11079:
	.loc 2 713 0
	movq	64(%r13), %rax
.LVL1859:
.LBE11079:
.LBE11078:
.LBB11081:
.LBB11051:
.LBB11048:
	.loc 2 738 0
	movq	-1768(%rbp), %rdi
.LBE11048:
.LBE11051:
.LBB11052:
.LBB11053:
	.loc 2 353 0
	andl	$4095, %ecx
.LBE11053:
.LBE11052:
.LBB11058:
.LBB11044:
	.loc 2 709 0
	movq	-1792(%rbp), %rsi
.LBE11044:
.LBE11058:
.LBE11081:
.LBB11082:
.LBB11080:
	.loc 2 713 0
	movl	(%rax), %edx
	movl	4(%rax), %eax
.LVL1860:
.LBE11080:
.LBE11082:
.LBB11083:
.LBB11059:
.LBB11049:
	.loc 2 738 0
	movq	%rdi, -712(%rbp)
.LBE11049:
.LBE11059:
.LBB11060:
.LBB11054:
	.loc 2 353 0
	leaq	-784(%rbp), %rdi
.LVL1861:
.LBE11054:
.LBE11060:
.LBB11061:
.LBB11045:
	.loc 2 709 0
	movq	%rsi, -720(%rbp)
.LVL1862:
.LBE11045:
.LBE11061:
.LBB11062:
.LBB11055:
	.loc 2 353 0
	movl	$2, %esi
.LBE11055:
.LBE11062:
.LBB11063:
.LBB11050:
	.loc 2 738 0
	movq	$0, -696(%rbp)
	movq	$0, -704(%rbp)
.LVL1863:
.LBE11050:
.LBE11063:
.LBB11064:
.LBB11056:
	.loc 2 352 0
	movl	%edx, -1600(%rbp)
	.loc 2 353 0
	leaq	-1600(%rbp), %rdx
.LBE11056:
.LBE11064:
.LBB11065:
.LBB11066:
	.loc 2 60 0
	movl	$1124007936, -784(%rbp)
	.loc 2 61 0
	movl	$0, -772(%rbp)
	movl	$0, -776(%rbp)
	movl	$0, -780(%rbp)
	.loc 2 62 0
	movq	$0, -736(%rbp)
	movq	$0, -744(%rbp)
	movq	$0, -752(%rbp)
	movq	$0, -768(%rbp)
	.loc 2 63 0
	movq	$0, -760(%rbp)
	.loc 2 64 0
	movq	$0, -728(%rbp)
.LVL1864:
.LBE11066:
.LBE11065:
.LBB11067:
.LBB11057:
	.loc 2 352 0
	movl	%eax, -1596(%rbp)
	.cfi_escape 0x2e,0
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL1865:
.LEHE9:
.LBE11057:
.LBE11067:
.LBE11083:
.LBB11084:
.LBB11085:
	.loc 15 780 0
	movslq	-1752(%rbp), %rax
.LVL1866:
	leaq	(%rax,%rax,2), %rbx
.LBE11085:
.LBE11084:
.LBB11087:
.LBB11088:
	.loc 2 283 0
	leaq	-784(%rbp), %rax
.LVL1867:
.LBE11088:
.LBE11087:
.LBB11101:
.LBB11086:
	.loc 15 780 0
	salq	$5, %rbx
	addq	-1744(%rbp), %rbx
.LVL1868:
.LBE11086:
.LBE11101:
.LBB11102:
.LBB11099:
	.loc 2 283 0
	cmpq	%rax, %rbx
	je	.L868
	.loc 2 285 0
	movq	-760(%rbp), %rax
.LVL1869:
	testq	%rax, %rax
	je	.L869
	.loc 2 286 0
	lock addl	$1, (%rax)
.L869:
.LVL1870:
.LBB11089:
.LBB11090:
	.loc 2 366 0
	movq	24(%rbx), %rax
	testq	%rax, %rax
	je	.L934
	lock subl	$1, (%rax)
	je	.L1032
.L934:
.LBB11091:
	.loc 2 369 0
	movl	4(%rbx), %eax
.LBE11091:
	.loc 2 368 0
	movq	$0, 48(%rbx)
	movq	$0, 40(%rbx)
	movq	$0, 32(%rbx)
	movq	$0, 16(%rbx)
.LVL1871:
.LBB11092:
	.loc 2 369 0
	testl	%eax, %eax
	jle	.L1033
	movq	64(%rbx), %rdx
	movl	-1752(%rbp), %edi
	xorl	%eax, %eax
.LVL1872:
	.p2align 4,,10
	.p2align 3
.L872:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	4(%rbx), %ecx
	addl	$1, %eax
.LVL1873:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L872
.LBE11092:
.LBE11090:
.LBE11089:
	.loc 2 288 0
	movl	-784(%rbp), %eax
.LVL1874:
	.loc 2 289 0
	cmpl	$2, %ecx
	movl	%edi, -1752(%rbp)
.LVL1875:
.LBB11096:
.LBB11093:
	.loc 2 371 0
	movq	$0, 24(%rbx)
.LVL1876:
.LBE11093:
.LBE11096:
	.loc 2 288 0
	movl	%eax, (%rbx)
	.loc 2 289 0
	jg	.L873
.L944:
	movl	-780(%rbp), %eax
	cmpl	$2, %eax
	jg	.L873
	.loc 2 291 0
	movl	%eax, 4(%rbx)
	.loc 2 292 0
	movl	-776(%rbp), %eax
	movq	-712(%rbp), %rdx
	movl	%eax, 8(%rbx)
	.loc 2 293 0
	movl	-772(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, 12(%rbx)
	movq	72(%rbx), %rax
.LVL1877:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL1878:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
.LVL1879:
.L874:
	.loc 2 299 0
	movdqa	-736(%rbp), %xmm0
	movdqa	-752(%rbp), %xmm1
	movdqa	-768(%rbp), %xmm2
	movups	%xmm1, 32(%rbx)
	movups	%xmm2, 16(%rbx)
	movups	%xmm0, 48(%rbx)
.L868:
.LVL1880:
.LBE11099:
.LBE11102:
.LBB11103:
.LBB11104:
.LBB11105:
.LBB11106:
	.loc 2 366 0
	movq	-760(%rbp), %rax
	testq	%rax, %rax
	je	.L937
	lock subl	$1, (%rax)
	jne	.L937
	.loc 2 367 0
	leaq	-784(%rbp), %rdi
.LVL1881:
	call	_ZN2cv3Mat10deallocateEv
.LVL1882:
.L937:
.LBB11107:
	.loc 2 369 0
	movl	-780(%rbp), %r9d
.LBE11107:
	.loc 2 368 0
	movq	$0, -736(%rbp)
	movq	$0, -744(%rbp)
	movq	$0, -752(%rbp)
	movq	$0, -768(%rbp)
.LVL1883:
.LBB11108:
	.loc 2 369 0
	testl	%r9d, %r9d
	jle	.L880
	movq	-720(%rbp), %rdx
	movl	-1752(%rbp), %esi
	xorl	%eax, %eax
.LVL1884:
	.p2align 4,,10
	.p2align 3
.L881:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL1885:
	addq	$4, %rdx
	cmpl	%eax, -780(%rbp)
	jg	.L881
	movl	%esi, -1752(%rbp)
.LVL1886:
.L880:
.LBE11108:
.LBE11106:
.LBE11105:
	.loc 2 277 0
	movq	-712(%rbp), %rdi
	cmpq	-1768(%rbp), %rdi
.LBB11110:
.LBB11109:
	.loc 2 371 0
	movq	$0, -760(%rbp)
.LVL1887:
.LBE11109:
.LBE11110:
	.loc 2 277 0
	je	.L879
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL1888:
.L879:
.LBE11104:
.LBE11103:
.LBB11111:
	.loc 1 855 0
	leaq	-1168(%rbp), %rax
	leaq	-1712(%rbp), %rbx
	xorl	%ecx, %ecx
	movl	$4, %edx
	movl	$_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.6, %edi
	movl	%r14d, -1676(%rbp)
	movq	%rax, -1696(%rbp)
	leaq	-1072(%rbp), %rax
	movq	%rbx, %rsi
	movb	%r12b, -1672(%rbp)
	movq	%r15, -1704(%rbp)
	movq	%rax, -1688(%rbp)
	movl	-1760(%rbp), %eax
	movq	%r13, -1712(%rbp)
	movl	%eax, -1680(%rbp)
	call	GOMP_parallel
.LVL1889:
.LBE11111:
	.loc 1 870 0
	cvttsd2si	16(%r15), %ecx
	cvttsd2si	8(%r15), %edx
	leaq	-976(%rbp), %rsi
	leaq	-1168(%rbp), %rdi
.LBB11112:
	.loc 1 855 0
	movl	-1680(%rbp), %r14d
.LVL1890:
	movl	-1676(%rbp), %r12d
.LVL1891:
.LEHB10:
.LBE11112:
	.loc 1 870 0
	call	_Z9boxFilterRKN2cv3MatERS0_ii
.LVL1892:
	.loc 1 871 0
	leaq	-880(%rbp), %rsi
	leaq	-1072(%rbp), %rdi
	cvttsd2si	16(%r15), %ecx
	cvttsd2si	8(%r15), %edx
	call	_Z9boxFilterRKN2cv3MatERS0_ii
.LVL1893:
.LEHE10:
.LBB11113:
	.loc 1 873 0
	leaq	-976(%rbp), %rax
	xorl	%ecx, %ecx
	movl	$4, %edx
	movq	%rbx, %rsi
	movl	$_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.7, %edi
	movl	%r14d, -1688(%rbp)
	movq	%rax, -1712(%rbp)
	leaq	-880(%rbp), %rax
	movl	%r12d, -1684(%rbp)
	movq	%rax, -1704(%rbp)
	movq	-1776(%rbp), %rax
	movq	%rax, -1696(%rbp)
	movl	-1752(%rbp), %eax
	movl	%eax, -1680(%rbp)
	call	GOMP_parallel
.LVL1894:
	movl	-1688(%rbp), %eax
	movl	-1684(%rbp), %r14d
.LVL1895:
	movl	%eax, -1760(%rbp)
.LVL1896:
.LBE11113:
.LBE11040:
	.loc 1 846 0
	movl	-1680(%rbp), %eax
.LVL1897:
	addl	$1, %eax
	cmpl	%eax, -1780(%rbp)
	movl	%eax, -1752(%rbp)
.LVL1898:
	jle	.L865
	movl	0(%r13), %ecx
	jmp	.L882
.LVL1899:
	.p2align 4,,10
	.p2align 3
.L873:
.LBB11115:
.LBB11114:
.LBB11100:
	.loc 2 298 0
	leaq	-784(%rbp), %rsi
	movq	%rbx, %rdi
.LEHB11:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL1900:
	jmp	.L874
.LVL1901:
	.p2align 4,,10
	.p2align 3
.L1032:
.LBB11097:
.LBB11094:
	.loc 2 367 0
	movq	%rbx, %rdi
	call	_ZN2cv3Mat10deallocateEv
.LVL1902:
.LEHE11:
	jmp	.L934
.LVL1903:
.L1033:
.LBE11094:
.LBE11097:
	.loc 2 288 0
	movl	-784(%rbp), %eax
.LBB11098:
.LBB11095:
	.loc 2 371 0
	movq	$0, 24(%rbx)
.LVL1904:
.LBE11095:
.LBE11098:
	.loc 2 288 0
	movl	%eax, (%rbx)
	jmp	.L944
.LVL1905:
.L1031:
	leaq	-1744(%rbp), %rax
.LVL1906:
	leaq	-1712(%rbp), %rbx
	movq	%rax, -1776(%rbp)
.LVL1907:
.L865:
.LBE11100:
.LBE11114:
.LBE11115:
.LBE11117:
.LBB11118:
	.loc 1 885 0
	movq	-1904(%rbp), %rax
	xorl	%ecx, %ecx
	movq	%rbx, %rsi
	movl	$4, %edx
	movl	$_ZN15BilateralFilter5applyERKN2cv3MatERS1_._omp_fn.8, %edi
	movq	%r15, -1696(%rbp)
	movq	%r13, -1712(%rbp)
	movl	%r14d, -1672(%rbp)
	movq	%rax, -1704(%rbp)
	movq	-1776(%rbp), %rax
	movq	%rax, -1688(%rbp)
	movl	-1780(%rbp), %eax
	movl	%eax, -1680(%rbp)
	movl	-1760(%rbp), %eax
	movl	%eax, -1676(%rbp)
	call	GOMP_parallel
.LVL1908:
.LBE11118:
.LBB11119:
.LBB11120:
	.loc 15 424 0
	movq	-1736(%rbp), %rbx
	movq	-1744(%rbp), %rax
.LVL1909:
.LBB11121:
.LBB11122:
.LBB11123:
.LBB11124:
.LBB11125:
	.loc 16 102 0
	cmpq	%rax, %rbx
	je	.L883
	movq	%rax, %r14
.LVL1910:
	.p2align 4,,10
	.p2align 3
.L891:
.LBB11126:
.LBB11127:
.LBB11128:
.LBB11129:
.LBB11130:
	.loc 2 366 0
	movq	24(%r14), %rax
.LVL1911:
	testq	%rax, %rax
	je	.L938
	lock subl	$1, (%rax)
	jne	.L938
	.loc 2 367 0
	movq	%r14, %rdi
	call	_ZN2cv3Mat10deallocateEv
.LVL1912:
.L938:
.LBB11131:
	.loc 2 369 0
	movl	4(%r14), %r8d
.LBE11131:
	.loc 2 368 0
	movq	$0, 48(%r14)
	movq	$0, 40(%r14)
	movq	$0, 32(%r14)
	movq	$0, 16(%r14)
.LVL1913:
.LBB11132:
	.loc 2 369 0
	testl	%r8d, %r8d
	jle	.L889
	movq	64(%r14), %rdx
	xorl	%eax, %eax
.LVL1914:
	.p2align 4,,10
	.p2align 3
.L890:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL1915:
	addq	$4, %rdx
	cmpl	%eax, 4(%r14)
	jg	.L890
.LVL1916:
.L889:
.LBE11132:
.LBE11130:
.LBE11129:
	.loc 2 277 0
	movq	72(%r14), %rdi
.LBB11134:
.LBB11133:
	.loc 2 371 0
	movq	%r14, %rax
	movq	$0, 24(%r14)
.LVL1917:
.LBE11133:
.LBE11134:
	.loc 2 277 0
	addq	$80, %rax
	cmpq	%rax, %rdi
	je	.L888
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL1918:
.L888:
.LBE11128:
.LBE11127:
.LBE11126:
	.loc 16 102 0
	movq	%r14, %rax
	addq	$96, %rax
	cmpq	%rax, %rbx
	movq	%rax, %r14
.LVL1919:
	jne	.L891
	movq	-1744(%rbp), %rbx
.LVL1920:
.L883:
.LBE11125:
.LBE11124:
.LBE11123:
.LBE11122:
.LBE11121:
.LBB11135:
.LBB11136:
.LBB11137:
	.loc 15 177 0
	testq	%rbx, %rbx
	je	.L892
.LVL1921:
.LBB11138:
.LBB11139:
.LBB11140:
	.loc 17 110 0
	movq	%rbx, %rdi
	call	_ZdlPv
.LVL1922:
.L892:
.LBE11140:
.LBE11139:
.LBE11138:
.LBE11137:
.LBE11136:
.LBE11135:
.LBE11120:
.LBE11119:
.LBB11141:
.LBB11142:
.LBB11143:
.LBB11144:
	.loc 2 366 0
	movq	-856(%rbp), %rax
	testq	%rax, %rax
	je	.L939
	lock subl	$1, (%rax)
	jne	.L939
	.loc 2 367 0
	leaq	-880(%rbp), %rdi
.LVL1923:
	call	_ZN2cv3Mat10deallocateEv
.LVL1924:
.L939:
.LBB11145:
	.loc 2 369 0
	movl	-876(%rbp), %edi
.LBE11145:
	.loc 2 368 0
	movq	$0, -832(%rbp)
	movq	$0, -840(%rbp)
	movq	$0, -848(%rbp)
	movq	$0, -864(%rbp)
.LVL1925:
.LBB11146:
	.loc 2 369 0
	testl	%edi, %edi
	jle	.L898
	movq	-816(%rbp), %rdx
	xorl	%eax, %eax
.LVL1926:
	.p2align 4,,10
	.p2align 3
.L899:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL1927:
	addq	$4, %rdx
	cmpl	%eax, -876(%rbp)
	jg	.L899
.LVL1928:
.L898:
.LBE11146:
.LBE11144:
.LBE11143:
	.loc 2 277 0
	movq	-808(%rbp), %rdi
	leaq	-880(%rbp), %rax
.LBB11148:
.LBB11147:
	.loc 2 371 0
	movq	$0, -856(%rbp)
.LVL1929:
.LBE11147:
.LBE11148:
	.loc 2 277 0
	addq	$80, %rax
	cmpq	%rax, %rdi
	je	.L897
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL1930:
.L897:
.LBE11142:
.LBE11141:
.LBB11149:
.LBB11150:
.LBB11151:
.LBB11152:
	.loc 2 366 0
	movq	-952(%rbp), %rax
	testq	%rax, %rax
	je	.L940
	lock subl	$1, (%rax)
	jne	.L940
	.loc 2 367 0
	leaq	-976(%rbp), %rdi
.LVL1931:
	call	_ZN2cv3Mat10deallocateEv
.LVL1932:
.L940:
.LBB11153:
	.loc 2 369 0
	movl	-972(%rbp), %esi
.LBE11153:
	.loc 2 368 0
	movq	$0, -928(%rbp)
	movq	$0, -936(%rbp)
	movq	$0, -944(%rbp)
	movq	$0, -960(%rbp)
.LVL1933:
.LBB11154:
	.loc 2 369 0
	testl	%esi, %esi
	jle	.L905
	movq	-912(%rbp), %rdx
	xorl	%eax, %eax
.LVL1934:
	.p2align 4,,10
	.p2align 3
.L906:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL1935:
	addq	$4, %rdx
	cmpl	%eax, -972(%rbp)
	jg	.L906
.LVL1936:
.L905:
.LBE11154:
.LBE11152:
.LBE11151:
	.loc 2 277 0
	movq	-904(%rbp), %rdi
	leaq	-976(%rbp), %rax
.LBB11156:
.LBB11155:
	.loc 2 371 0
	movq	$0, -952(%rbp)
.LVL1937:
.LBE11155:
.LBE11156:
	.loc 2 277 0
	addq	$80, %rax
	cmpq	%rax, %rdi
	je	.L904
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL1938:
.L904:
.LBE11150:
.LBE11149:
.LBB11157:
.LBB11158:
.LBB11159:
.LBB11160:
	.loc 2 366 0
	movq	-1048(%rbp), %rax
	testq	%rax, %rax
	je	.L941
	lock subl	$1, (%rax)
	jne	.L941
	.loc 2 367 0
	leaq	-1072(%rbp), %rdi
.LVL1939:
	call	_ZN2cv3Mat10deallocateEv
.LVL1940:
.L941:
.LBB11161:
	.loc 2 369 0
	movl	-1068(%rbp), %ecx
.LBE11161:
	.loc 2 368 0
	movq	$0, -1024(%rbp)
	movq	$0, -1032(%rbp)
	movq	$0, -1040(%rbp)
	movq	$0, -1056(%rbp)
.LVL1941:
.LBB11162:
	.loc 2 369 0
	testl	%ecx, %ecx
	jle	.L912
	movq	-1008(%rbp), %rdx
	xorl	%eax, %eax
.LVL1942:
	.p2align 4,,10
	.p2align 3
.L913:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL1943:
	addq	$4, %rdx
	cmpl	%eax, -1068(%rbp)
	jg	.L913
.LVL1944:
.L912:
.LBE11162:
.LBE11160:
.LBE11159:
	.loc 2 277 0
	movq	-1000(%rbp), %rdi
	leaq	-1072(%rbp), %rax
.LBB11164:
.LBB11163:
	.loc 2 371 0
	movq	$0, -1048(%rbp)
.LVL1945:
.LBE11163:
.LBE11164:
	.loc 2 277 0
	addq	$80, %rax
	cmpq	%rax, %rdi
	je	.L911
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL1946:
.L911:
.LBE11158:
.LBE11157:
.LBB11165:
.LBB11166:
.LBB11167:
.LBB11168:
	.loc 2 366 0
	movq	-1144(%rbp), %rax
	testq	%rax, %rax
	je	.L942
	lock subl	$1, (%rax)
	jne	.L942
	.loc 2 367 0
	leaq	-1168(%rbp), %rdi
.LVL1947:
	call	_ZN2cv3Mat10deallocateEv
.LVL1948:
.L942:
.LBB11169:
	.loc 2 369 0
	movl	-1164(%rbp), %edx
.LBE11169:
	.loc 2 368 0
	movq	$0, -1120(%rbp)
	movq	$0, -1128(%rbp)
	movq	$0, -1136(%rbp)
	movq	$0, -1152(%rbp)
.LVL1949:
.LBB11170:
	.loc 2 369 0
	testl	%edx, %edx
	jle	.L919
	movq	-1104(%rbp), %rdx
	xorl	%eax, %eax
.LVL1950:
	.p2align 4,,10
	.p2align 3
.L920:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL1951:
	addq	$4, %rdx
	cmpl	%eax, -1164(%rbp)
	jg	.L920
.LVL1952:
.L919:
.LBE11170:
.LBE11168:
.LBE11167:
	.loc 2 277 0
	movq	-1096(%rbp), %rdi
	leaq	-1168(%rbp), %rax
.LBB11172:
.LBB11171:
	.loc 2 371 0
	movq	$0, -1144(%rbp)
.LVL1953:
.LBE11171:
.LBE11172:
	.loc 2 277 0
	addq	$80, %rax
	cmpq	%rax, %rdi
	je	.L918
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL1954:
.L918:
.LBE11166:
.LBE11165:
.LBB11173:
.LBB11174:
.LBB11175:
.LBB11176:
	.loc 15 177 0
	movq	-1880(%rbp), %rax
	testq	%rax, %rax
	je	.L833
.LVL1955:
.LBB11177:
.LBB11178:
.LBB11179:
	.loc 17 110 0
	movq	%rax, %rdi
	call	_ZdlPv
.LVL1956:
.L833:
.LBE11179:
.LBE11178:
.LBE11177:
.LBE11176:
.LBE11175:
.LBE11174:
.LBE11173:
	.loc 1 906 0
	movq	-56(%rbp), %rax
	xorq	%fs:40, %rax
	jne	.L1034
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
.LVL1957:
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
.LVL1958:
	ret
.LVL1959:
.L1026:
	.cfi_restore_state
.LBB11180:
.LBB11034:
.LBB11029:
.LBB11019:
.LBB11004:
.LBB11005:
	.loc 5 53 0
	movq	-672(%rbp), %rax
	movq	-664(%rbp), %rdx
	movq	%rax, -624(%rbp)
	movq	%rdx, -616(%rbp)
.LVL1960:
.LBE11005:
.LBE11004:
.LBB11006:
.LBB11007:
	movq	-656(%rbp), %rax
	movq	-648(%rbp), %rdx
	movq	%rax, -608(%rbp)
	movq	%rdx, -600(%rbp)
.LVL1961:
.LBE11007:
.LBE11006:
.LBB11008:
.LBB11009:
	movq	-640(%rbp), %rax
	movq	-632(%rbp), %rdx
	movq	%rax, -592(%rbp)
	movq	%rdx, -584(%rbp)
	jmp	.L853
.LVL1962:
.L835:
.LBE11009:
.LBE11008:
.LBE11019:
.LBE11029:
.LBE11034:
.LBE11180:
.LBB11181:
.LBB10287:
.LBB10257:
.LBB10252:
	.loc 15 187 0
	movq	$0, -1728(%rbp)
.LVL1963:
.LBB10249:
.LBB10246:
	.loc 15 170 0
	xorl	%eax, %eax
	jmp	.L943
.LVL1964:
.L1034:
.LBE10246:
.LBE10249:
.LBE10252:
.LBE10257:
.LBE10287:
.LBE11181:
	.loc 1 906 0
	call	__stack_chk_fail
.LVL1965:
.L1025:
	leaq	-1744(%rbp), %rax
.LVL1966:
.LBB11182:
.LBB11035:
.LBB11030:
.LBB11020:
.LBB10636:
.LBB10623:
.LBB10624:
	.loc 8 98 0
	movsd	-1808(%rbp), %xmm0
	movq	%rax, -1776(%rbp)
.LVL1967:
.L1013:
.LBE10624:
.LBE10623:
.LBE10636:
.LBE11020:
.LBB11021:
.LBB11010:
.LBB11001:
.LBB10990:
.LBB10991:
	movq	stderr(%rip), %rdi
	movl	$.LC37, %edx
	movl	$1, %esi
	movl	$1, %eax
.LEHB12:
	call	__fprintf_chk
.LVL1968:
.L849:
.LBE10991:
.LBE10990:
.LBE11001:
.LBE11010:
.LBE11021:
.LBB11022:
.LBB10637:
	.loc 1 661 0
	movl	$-1, %edi
	call	exit
.LVL1969:
.L1023:
	leaq	-1744(%rbp), %rax
.LVL1970:
.LBB10625:
.LBB10626:
	.loc 8 98 0
	movsd	-1808(%rbp), %xmm0
	movq	%rax, -1776(%rbp)
.LVL1971:
.L1015:
	movq	stderr(%rip), %rdi
	movl	$.LC35, %edx
	movl	$1, %esi
	movl	$1, %eax
	call	__fprintf_chk
.LVL1972:
	jmp	.L849
.LVL1973:
.L1024:
	leaq	-1744(%rbp), %rax
.LVL1974:
.LBE10626:
.LBE10625:
.LBB10627:
.LBB10628:
	movsd	-1808(%rbp), %xmm0
	movq	%rax, -1776(%rbp)
.LVL1975:
.L1011:
.LBE10628:
.LBE10627:
.LBE10637:
.LBE11022:
.LBB11023:
.LBB11011:
.LBB11002:
.LBB10993:
.LBB10994:
	movq	stderr(%rip), %rdi
	movl	$.LC36, %edx
	movl	$1, %esi
	movl	$1, %eax
	call	__fprintf_chk
.LVL1976:
.LEHE12:
	jmp	.L849
.LVL1977:
.L840:
.LBE10994:
.LBE10993:
.LBE11002:
.LBE11011:
.LBE11023:
.LBB11024:
.LBB10638:
.LBB10629:
.LBB10455:
.LBB10451:
.LBB10441:
.LBB10430:
.LBB10428:
.LBB10426:
	.loc 14 1323 0
	movapd	%xmm4, %xmm0
	movapd	%xmm5, %xmm1
	movsd	-488(%rbp), %xmm3
	movsd	-496(%rbp), %xmm2
	call	__muldc3
.LVL1978:
.LBE10426:
.LBE10428:
.LBE10430:
.LBE10441:
	.loc 1 649 0
	movsd	%xmm0, -480(%rbp)
	movq	%r13, -1776(%rbp)
	movsd	%xmm1, -472(%rbp)
	movsd	-16(%r13), %xmm4
	movsd	-8(%r13), %xmm6
	movsd	8(%r13), %xmm7
	movsd	%xmm4, -1760(%rbp)
	movsd	%xmm6, -1768(%rbp)
	movsd	%xmm7, -1792(%rbp)
	jmp	.L931
.LVL1979:
.L841:
.LBB10442:
.LBB10393:
.LBB10386:
.LBB10379:
	.loc 14 1323 0
	movq	-1776(%rbp), %rax
	movsd	-424(%rbp), %xmm3
	movsd	-432(%rbp), %xmm2
	movsd	8(%rax), %xmm1
	movsd	(%rax), %xmm0
	call	__muldc3
.LVL1980:
.LBE10379:
.LBE10386:
.LBE10393:
.LBE10442:
	.loc 1 647 0
	movsd	%xmm0, -400(%rbp)
	movsd	%xmm1, -392(%rbp)
	movsd	-16(%r13), %xmm3
	movsd	-8(%r13), %xmm5
	movsd	%xmm3, -1760(%rbp)
	movsd	%xmm5, -1768(%rbp)
	jmp	.L842
.LVL1981:
.L952:
	movq	%rax, %rbx
.L923:
.LBE10451:
.LBE10455:
.LBE10629:
.LBE10638:
.LBE11024:
.LBE11030:
.LBE11035:
.LBE11182:
	.loc 1 839 0
	movq	-1776(%rbp), %rdi
	call	_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev
.LVL1982:
.L924:
	.loc 1 838 0
	leaq	-880(%rbp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL1983:
.L925:
	.loc 1 837 0
	leaq	-976(%rbp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL1984:
.L926:
	.loc 1 836 0
	leaq	-1072(%rbp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL1985:
.L927:
	.loc 1 835 0
	leaq	-1168(%rbp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL1986:
.L928:
.LBB11183:
.LBB11184:
.LBB11185:
.LBB11186:
	.loc 15 177 0
	movq	-1880(%rbp), %rax
	testq	%rax, %rax
	je	.L929
.LVL1987:
.LBB11187:
.LBB11188:
.LBB11189:
	.loc 17 110 0
	movq	%rax, %rdi
	call	_ZdlPv
.LVL1988:
.L929:
	movq	%rbx, %rdi
.LEHB13:
	call	_Unwind_Resume
.LVL1989:
.LEHE13:
.L1022:
	leaq	-1744(%rbp), %rax
.LVL1990:
.LBE11189:
.LBE11188:
.LBE11187:
.LBE11186:
.LBE11185:
.LBE11184:
.LBE11183:
.LBB11190:
.LBB11036:
.LBB11031:
.LBB11025:
.LBB10639:
	movsd	-1808(%rbp), %xmm0
	movq	%rax, -1776(%rbp)
.LEHB14:
	call	_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_.part.37
.LVL1991:
.LEHE14:
.L947:
	movq	%rax, %rbx
	jmp	.L928
.LVL1992:
.L948:
	movq	%rax, %rbx
	jmp	.L927
.LVL1993:
.L949:
	movq	%rax, %rbx
	jmp	.L926
.LVL1994:
.L950:
	movq	%rax, %rbx
	jmp	.L925
.LVL1995:
.L951:
	movq	%rax, %rbx
.LVL1996:
	jmp	.L924
.LVL1997:
.L839:
.LBB10630:
.LBB10456:
.LBB10452:
.LBB10443:
.LBB10394:
.LBB10387:
.LBB10380:
	.loc 14 1323 0
	movsd	-568(%rbp), %xmm3
	movsd	-576(%rbp), %xmm2
	call	__muldc3
.LVL1998:
.LBE10380:
.LBE10387:
.LBE10394:
.LBE10443:
	.loc 1 647 0
	leaq	-576(%rbp), %rax
	movsd	8(%rbx), %xmm5
	movsd	%xmm0, -560(%rbp)
	addq	$16, %rax
	movsd	%xmm1, -552(%rbp)
	movsd	%xmm5, -1752(%rbp)
	movq	%rax, -1776(%rbp)
	jmp	.L930
.LVL1999:
.L1021:
.LEHB15:
.LBE10452:
.LBE10456:
.LBE10630:
.LBE10639:
.LBE11025:
.LBE11031:
.LBE11036:
.LBE11190:
.LBB11191:
.LBB10288:
.LBB10258:
.LBB10253:
.LBB10250:
.LBB10247:
.LBB10244:
.LBB10243:
.LBB10242:
	.loc 17 102 0
	call	_ZSt17__throw_bad_allocv
.LVL2000:
.LEHE15:
.L953:
.LBE10242:
.LBE10243:
.LBE10244:
.LBE10247:
.LBE10250:
.LBE10253:
.LBE10258:
.LBE10288:
.LBE11191:
.LBB11192:
.LBB11116:
	.loc 1 848 0
	leaq	-784(%rbp), %rdi
	movq	%rax, %rbx
.LVL2001:
	call	_ZN2cv3MatD1Ev
.LVL2002:
	jmp	.L923
.LVL2003:
.L854:
.LBE11116:
.LBE11192:
.LBB11193:
.LBB11037:
.LBB11032:
.LBB11026:
.LBB11012:
.LBB11003:
.LBB10996:
.LBB10820:
.LBB10816:
.LBB10804:
.LBB10747:
.LBB10740:
.LBB10733:
	.loc 14 1323 0
	movapd	%xmm0, %xmm2
	movsd	-312(%rbp), %xmm1
.LVL2004:
	movsd	-320(%rbp), %xmm0
.LVL2005:
	call	__muldc3
.LVL2006:
	movsd	8(%rbx), %xmm4
.LBE10733:
.LBE10740:
.LBE10747:
.LBE10804:
	.loc 1 647 0
	leaq	-320(%rbp), %rax
	movsd	%xmm0, -304(%rbp)
	movsd	%xmm1, -296(%rbp)
	leaq	16(%rax), %r15
	movsd	%xmm4, -1752(%rbp)
	jmp	.L932
.LVL2007:
.L855:
.LBB10805:
.LBB10789:
.LBB10785:
.LBB10781:
	.loc 14 1323 0
	movsd	-232(%rbp), %xmm3
.LBE10781:
.LBE10785:
.LBE10789:
.LBE10805:
	.loc 1 649 0
	movq	%r14, %r15
.LVL2008:
.LBB10806:
.LBB10790:
.LBB10786:
.LBB10782:
	.loc 14 1323 0
	movapd	%xmm5, %xmm1
	movapd	%xmm4, %xmm0
	movsd	-240(%rbp), %xmm2
	call	__muldc3
.LVL2009:
.LBE10782:
.LBE10786:
.LBE10790:
.LBE10806:
	.loc 1 649 0
	movsd	%xmm0, -224(%rbp)
	movsd	%xmm1, -216(%rbp)
	movsd	-16(%r14), %xmm6
	movsd	-8(%r14), %xmm7
	movsd	8(%r14), %xmm3
	movsd	%xmm6, -1768(%rbp)
	movsd	%xmm7, -1776(%rbp)
	movsd	%xmm3, -1792(%rbp)
	jmp	.L933
.LVL2010:
.L856:
.LBB10807:
.LBB10748:
.LBB10741:
.LBB10734:
	.loc 14 1323 0
	movsd	8(%r15), %xmm1
	movsd	(%r15), %xmm0
	movsd	-168(%rbp), %xmm3
	movsd	-176(%rbp), %xmm2
	call	__muldc3
.LVL2011:
.LBE10734:
.LBE10741:
.LBE10748:
.LBE10807:
	.loc 1 647 0
	movsd	%xmm0, -144(%rbp)
	movsd	%xmm1, -136(%rbp)
	movsd	-16(%r14), %xmm5
	movsd	-8(%r14), %xmm4
	movsd	%xmm5, -1768(%rbp)
	movsd	%xmm4, -1776(%rbp)
	jmp	.L857
.LVL2012:
.L1027:
	leaq	-1744(%rbp), %rax
.LVL2013:
.LBE10816:
.LBE10820:
.LBE10996:
	movsd	-1800(%rbp), %xmm0
	movq	%rax, -1776(%rbp)
.LEHB16:
	call	_ZN15fastGaussianIIR16getCoefficients_EdPfS0_S0_.part.37
.LVL2014:
.LEHE16:
.L1028:
	leaq	-1744(%rbp), %rax
.LVL2015:
.LBB10997:
.LBB10998:
	.loc 8 98 0
	movsd	-1800(%rbp), %xmm0
	movq	%rax, -1776(%rbp)
	jmp	.L1015
.LVL2016:
.L1029:
	leaq	-1744(%rbp), %rax
.LVL2017:
.LBE10998:
.LBE10997:
.LBB10999:
.LBB10995:
	movsd	-1800(%rbp), %xmm0
	movq	%rax, -1776(%rbp)
	jmp	.L1011
.LVL2018:
.L1030:
	leaq	-1744(%rbp), %rax
.LVL2019:
.LBE10995:
.LBE10999:
.LBB11000:
.LBB10992:
	movsd	-1800(%rbp), %xmm0
	movq	%rax, -1776(%rbp)
	jmp	.L1013
.LBE10992:
.LBE11000:
.LBE11003:
.LBE11012:
.LBE11026:
.LBE11032:
.LBE11037:
.LBE11193:
	.cfi_endproc
.LFE11298:
	.section	.gcc_except_table
.LLSDA11298:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE11298-.LLSDACSB11298
.LLSDACSB11298:
	.uleb128 .LEHB3-.LFB11298
	.uleb128 .LEHE3-.LEHB3
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB4-.LFB11298
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L947-.LFB11298
	.uleb128 0
	.uleb128 .LEHB5-.LFB11298
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L948-.LFB11298
	.uleb128 0
	.uleb128 .LEHB6-.LFB11298
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L949-.LFB11298
	.uleb128 0
	.uleb128 .LEHB7-.LFB11298
	.uleb128 .LEHE7-.LEHB7
	.uleb128 .L950-.LFB11298
	.uleb128 0
	.uleb128 .LEHB8-.LFB11298
	.uleb128 .LEHE8-.LEHB8
	.uleb128 .L951-.LFB11298
	.uleb128 0
	.uleb128 .LEHB9-.LFB11298
	.uleb128 .LEHE9-.LEHB9
	.uleb128 .L952-.LFB11298
	.uleb128 0
	.uleb128 .LEHB10-.LFB11298
	.uleb128 .LEHE10-.LEHB10
	.uleb128 .L952-.LFB11298
	.uleb128 0
	.uleb128 .LEHB11-.LFB11298
	.uleb128 .LEHE11-.LEHB11
	.uleb128 .L953-.LFB11298
	.uleb128 0
	.uleb128 .LEHB12-.LFB11298
	.uleb128 .LEHE12-.LEHB12
	.uleb128 .L952-.LFB11298
	.uleb128 0
	.uleb128 .LEHB13-.LFB11298
	.uleb128 .LEHE13-.LEHB13
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB14-.LFB11298
	.uleb128 .LEHE14-.LEHB14
	.uleb128 .L952-.LFB11298
	.uleb128 0
	.uleb128 .LEHB15-.LFB11298
	.uleb128 .LEHE15-.LEHB15
	.uleb128 .L951-.LFB11298
	.uleb128 0
	.uleb128 .LEHB16-.LFB11298
	.uleb128 .LEHE16-.LEHB16
	.uleb128 .L952-.LFB11298
	.uleb128 0
.LLSDACSE11298:
	.text
	.size	_ZN15BilateralFilter5applyERKN2cv3MatERS1_, .-_ZN15BilateralFilter5applyERKN2cv3MatERS1_
	.section	.text.unlikely
.LCOLDE52:
	.text
.LHOTE52:
	.section	.rodata.str1.8
	.align 8
.LC53:
	.string	"Usage: inputfile outputfile sigma rangesigma factor\n"
	.section	.rodata.str1.1
.LC54:
	.string	"["
.LC55:
	.string	" x "
.LC56:
	.string	"]"
	.section	.rodata.str1.8
	.align 8
.LC57:
	.string	"clock cycles of fastGaussianIIR(): "
	.align 8
.LC58:
	.string	"elapsed time of fastGaussianIIR(): "
	.section	.rodata.str1.1
.LC59:
	.string	"s\n"
	.section	.rodata.str1.8
	.align 8
.LC61:
	.string	"clock cycles of opencv::GaussianBlur(): "
	.align 8
.LC62:
	.string	"elapsed time of opencv::GaussianBlur(): "
	.section	.rodata.str1.1
.LC63:
	.string	"gaussian_blur.jpg"
.LC64:
	.string	"error.jpg"
.LC65:
	.string	"blf.jpg"
	.section	.text.unlikely
.LCOLDB66:
	.section	.text.startup,"ax",@progbits
.LHOTB66:
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB11299:
	.loc 1 909 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA11299
.LVL2020:
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	subq	$3168, %rsp
	.cfi_def_cfa_offset 3200
	.loc 1 909 0
	movq	%fs:40, %rax
	movq	%rax, 3160(%rsp)
	xorl	%eax, %eax
	.loc 1 910 0
	cmpl	$5, %edi
	jle	.L1116
.LBB11560:
.LBB11561:
	.file 19 "/usr/include/x86_64-linux-gnu/bits/stdlib-float.h"
	.loc 19 28 0
	movq	24(%rsi), %rdi
.LVL2021:
	movq	%rsi, %rbx
.LBE11561:
.LBE11560:
	.loc 1 914 0
	movq	8(%rsi), %r12
.LVL2022:
	.loc 1 915 0
	movq	16(%rsi), %rbp
.LVL2023:
.LBB11563:
.LBB11562:
	.loc 19 28 0
	xorl	%esi, %esi
.LVL2024:
	call	strtod
.LVL2025:
.LBE11562:
.LBE11563:
	.loc 1 916 0
	pxor	%xmm3, %xmm3
.LBB11564:
.LBB11565:
	.loc 19 28 0
	movq	32(%rbx), %rdi
	xorl	%esi, %esi
.LBE11565:
.LBE11564:
	.loc 1 916 0
	cvtsd2ss	%xmm0, %xmm3
	movss	%xmm3, (%rsp)
.LVL2026:
.LBB11567:
.LBB11566:
	.loc 19 28 0
	call	strtod
.LVL2027:
.LBE11566:
.LBE11567:
	.loc 1 917 0
	pxor	%xmm1, %xmm1
.LBB11568:
.LBB11569:
	.loc 19 28 0
	movq	40(%rbx), %rdi
	xorl	%esi, %esi
.LBE11569:
.LBE11568:
	.loc 1 917 0
	cvtsd2ss	%xmm0, %xmm1
	movss	%xmm1, 24(%rsp)
.LVL2028:
.LBB11571:
.LBB11570:
	.loc 19 28 0
	call	strtod
.LVL2029:
.LBE11570:
.LBE11571:
	.loc 1 919 0
	leaq	1088(%rsp), %rdi
	movq	%r12, %rsi
.LEHB17:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_.isra.64
.LVL2030:
.LEHE17:
	leaq	1088(%rsp), %rsi
	leaq	240(%rsp), %rdi
	movl	$1, %edx
.LEHB18:
	call	_ZN2cv6imreadERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEi
.LVL2031:
.LEHE18:
	movq	1088(%rsp), %rdi
.LVL2032:
.LBB11572:
.LBB11573:
.LBB11574:
	.loc 9 179 0
	leaq	1104(%rsp), %rax
	cmpq	%rax, %rdi
	je	.L1038
.LVL2033:
.LBB11575:
.LBB11576:
.LBB11577:
.LBB11578:
	.loc 17 110 0
	call	_ZdlPv
.LVL2034:
.L1038:
.LBE11578:
.LBE11577:
.LBE11576:
.LBE11575:
.LBE11574:
.LBE11573:
.LBE11572:
.LBB11579:
.LBB11580:
	.loc 2 713 0
	movq	304(%rsp), %rax
.LBE11580:
.LBE11579:
.LBB11582:
.LBB11583:
.LBB11584:
.LBB11585:
	.loc 12 561 0
	movl	$1, %edx
	movl	$.LC54, %esi
	movl	$_ZSt4cout, %edi
.LBE11585:
.LBE11584:
.LBE11583:
.LBE11582:
.LBB11595:
.LBB11581:
	.loc 2 713 0
	movl	(%rax), %r12d
.LVL2035:
	movl	4(%rax), %ebx
.LVL2036:
.LEHB19:
.LBE11581:
.LBE11595:
.LBB11596:
.LBB11594:
.LBB11587:
.LBB11586:
	.loc 12 561 0
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.LVL2037:
.LBE11586:
.LBE11587:
	.loc 13 4034 0
	movl	%ebx, %esi
	movl	$_ZSt4cout, %edi
	call	_ZNSolsEi
.LVL2038:
.LBB11588:
.LBB11589:
	.loc 12 561 0
	movl	$3, %edx
	movl	$.LC55, %esi
	movq	%rax, %rdi
.LBE11589:
.LBE11588:
	.loc 13 4034 0
	movq	%rax, %rbx
.LVL2039:
.LBB11591:
.LBB11590:
	.loc 12 561 0
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.LVL2040:
.LBE11590:
.LBE11591:
	.loc 13 4034 0
	movl	%r12d, %esi
	movq	%rbx, %rdi
	call	_ZNSolsEi
.LVL2041:
.LBB11592:
.LBB11593:
	.loc 12 561 0
	movl	$1, %edx
	movl	$.LC56, %esi
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.LVL2042:
.LBE11593:
.LBE11592:
.LBE11594:
.LBE11596:
	.loc 1 920 0
	movl	$_ZSt4cout, %edi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.constprop.80
.LVL2043:
.LBB11597:
.LBB11598:
	.loc 2 713 0 discriminator 5
	movq	304(%rsp), %rax
.LBE11598:
.LBE11597:
.LBB11605:
.LBB11606:
	.loc 2 400 0 discriminator 5
	movl	240(%rsp), %edx
.LBE11606:
.LBE11605:
	.loc 1 922 0 discriminator 5
	leaq	80(%rsp), %rsi
	leaq	336(%rsp), %rdi
.LBB11608:
.LBB11603:
	.loc 2 713 0 discriminator 5
	movl	(%rax), %ecx
.LBB11599:
.LBB11600:
	.loc 13 1861 0 discriminator 5
	movl	4(%rax), %eax
.LBE11600:
.LBE11599:
.LBE11603:
.LBE11608:
.LBB11609:
.LBB11607:
	.loc 2 400 0 discriminator 5
	andl	$4095, %edx
.LVL2044:
.LBE11607:
.LBE11609:
.LBB11610:
.LBB11604:
.LBB11602:
.LBB11601:
	.loc 13 1861 0 discriminator 5
	movl	%eax, 80(%rsp)
	movl	%ecx, 84(%rsp)
.LVL2045:
.LBE11601:
.LBE11602:
.LBE11604:
.LBE11610:
	.loc 1 922 0 discriminator 5
	call	_ZN2cv3MatC1ENS_5Size_IiEEi
.LVL2046:
.LEHE19:
	.loc 1 923 0 discriminator 3
	pxor	%xmm4, %xmm4
.LBB11611:
.LBB11612:
	.loc 1 338 0 discriminator 3
	leaq	624(%rsp), %rdi
.LBE11612:
.LBE11611:
	.loc 1 923 0 discriminator 3
	cvtss2sd	(%rsp), %xmm4
	movsd	%xmm4, 8(%rsp)
.LVL2047:
.LBB11614:
.LBB11613:
	.loc 1 336 0 discriminator 3
	movsd	%xmm4, 624(%rsp)
	movsd	%xmm4, 632(%rsp)
.LEHB20:
	.loc 1 338 0 discriminator 3
	call	_ZN15fastGaussianIIR15getCoefficientsEv
.LVL2048:
.LBE11613:
.LBE11614:
	.loc 1 926 0
	call	clock
.LVL2049:
	movq	%rax, %rbx
.LVL2050:
	.loc 1 927 0
	call	omp_get_wtime
.LVL2051:
	.loc 1 930 0
	leaq	336(%rsp), %rsi
	leaq	240(%rsp), %rdi
.LVL2052:
	movl	$30, %ecx
	movl	$30, %edx
	.loc 1 927 0
	movsd	%xmm0, 16(%rsp)
.LVL2053:
	.loc 1 930 0
	call	_Z9boxFilterRKN2cv3MatERS0_ii
.LVL2054:
	.loc 1 931 0
	call	omp_get_wtime
.LVL2055:
	movapd	%xmm0, %xmm5
	subsd	16(%rsp), %xmm5
	movsd	%xmm5, 16(%rsp)
.LVL2056:
	.loc 1 932 0
	call	clock
.LVL2057:
	.loc 1 933 0
	pxor	%xmm7, %xmm7
	subq	%rbx, %rax
.LVL2058:
.LBB11615:
.LBB11616:
	.loc 12 561 0
	movl	$35, %edx
	movl	$.LC57, %esi
	movl	$_ZSt4cout, %edi
.LBE11616:
.LBE11615:
	.loc 1 933 0
	cvtsi2ssq	%rax, %xmm7
	movss	%xmm7, 28(%rsp)
.LBB11618:
.LBB11617:
	.loc 12 561 0
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.LVL2059:
.LBE11617:
.LBE11618:
.LBB11619:
.LBB11620:
	.loc 12 228 0
	pxor	%xmm0, %xmm0
	movl	$_ZSt4cout, %edi
	cvtss2sd	28(%rsp), %xmm0
	call	_ZNSo9_M_insertIdEERSoT_
.LVL2060:
.LBE11620:
.LBE11619:
	.loc 1 933 0
	movq	%rax, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.constprop.80
.LVL2061:
.LBB11621:
.LBB11622:
	.loc 12 561 0
	movl	$35, %edx
	movl	$.LC58, %esi
	movl	$_ZSt4cout, %edi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.LVL2062:
.LBE11622:
.LBE11621:
.LBB11623:
.LBB11624:
	.loc 12 221 0
	movsd	16(%rsp), %xmm0
	movl	$_ZSt4cout, %edi
	call	_ZNSo9_M_insertIdEERSoT_
.LVL2063:
.LBE11624:
.LBE11623:
	.loc 1 934 0
	movl	$.LC59, %esi
	movq	%rax, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.LVL2064:
.LEHE20:
	.loc 1 935 0
	leaq	336(%rsp), %rsi
	leaq	80(%rsp), %rdi
.LBB11625:
.LBB11626:
.LBB11627:
.LBB11628:
	.loc 15 87 0
	movq	$0, 48(%rsp)
	movq	$0, 56(%rsp)
	movq	$0, 64(%rsp)
.LVL2065:
.LEHB21:
.LBE11628:
.LBE11627:
.LBE11626:
.LBE11625:
	.loc 1 935 0
	call	_ZN2cv11_InputArrayC1ERKNS_3MatE
.LVL2066:
	.loc 1 935 0 is_stmt 0 discriminator 2
	leaq	112(%rsp), %rdi
	movq	%rbp, %rsi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_.isra.64
.LVL2067:
.LEHE21:
	.loc 1 935 0 discriminator 4
	leaq	48(%rsp), %rdx
	leaq	80(%rsp), %rsi
	leaq	112(%rsp), %rdi
.LEHB22:
	call	_ZN2cv7imwriteERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKNS_11_InputArrayERKSt6vectorIiSaIiEE
.LVL2068:
.LEHE22:
	movq	112(%rsp), %rdi
.LVL2069:
.LBB11629:
.LBB11630:
.LBB11631:
	.loc 9 179 0 is_stmt 1
	leaq	128(%rsp), %rax
	cmpq	%rax, %rdi
	je	.L1039
.LVL2070:
.LBB11632:
.LBB11633:
.LBB11634:
.LBB11635:
	.loc 17 110 0
	call	_ZdlPv
.LVL2071:
.L1039:
.LBE11635:
.LBE11634:
.LBE11633:
.LBE11632:
.LBE11631:
.LBE11630:
.LBE11629:
.LBB11636:
.LBB11637:
.LBB11638:
	.loc 15 161 0
	movq	48(%rsp), %rdi
.LVL2072:
.LBB11639:
.LBB11640:
	.loc 15 177 0
	testq	%rdi, %rdi
	je	.L1040
.LVL2073:
.LBB11641:
.LBB11642:
.LBB11643:
	.loc 17 110 0
	call	_ZdlPv
.LVL2074:
.L1040:
.LBE11643:
.LBE11642:
.LBE11641:
.LBE11640:
.LBE11639:
.LBE11638:
.LBE11637:
.LBE11636:
.LBB11644:
.LBB11645:
	.loc 2 713 0
	movq	304(%rsp), %rax
.LBE11645:
.LBE11644:
.LBB11652:
.LBB11653:
	.loc 2 400 0
	movl	240(%rsp), %edx
.LBE11653:
.LBE11652:
	.loc 1 937 0
	leaq	80(%rsp), %rsi
	leaq	432(%rsp), %rdi
.LBB11655:
.LBB11650:
	.loc 2 713 0
	movl	(%rax), %ecx
.LBB11646:
.LBB11647:
	.loc 13 1861 0
	movl	4(%rax), %eax
.LBE11647:
.LBE11646:
.LBE11650:
.LBE11655:
.LBB11656:
.LBB11654:
	.loc 2 400 0
	andl	$4095, %edx
.LVL2075:
.LBE11654:
.LBE11656:
.LBB11657:
.LBB11651:
.LBB11649:
.LBB11648:
	.loc 13 1861 0
	movl	%eax, 80(%rsp)
	movl	%ecx, 84(%rsp)
.LVL2076:
.LEHB23:
.LBE11648:
.LBE11649:
.LBE11651:
.LBE11657:
	.loc 1 937 0
	call	_ZN2cv3MatC1ENS_5Size_IiEEi
.LVL2077:
.LEHE23:
	.loc 1 938 0 discriminator 3
	call	clock
.LVL2078:
	movq	%rax, %rbx
.LVL2079:
	.loc 1 939 0 discriminator 3
	call	omp_get_wtime
.LVL2080:
	.loc 1 940 0 discriminator 3
	movss	(%rsp), %xmm6
	leaq	432(%rsp), %rsi
	mulss	.LC60(%rip), %xmm6
	leaq	80(%rsp), %rdi
	.loc 1 939 0 discriminator 3
	movsd	%xmm0, 16(%rsp)
.LVL2081:
	.loc 1 940 0 discriminator 3
	cvttss2si	%xmm6, %eax
	leal	1(%rax,%rax), %eax
.LVL2082:
.LBB11658:
.LBB11659:
	.loc 13 1861 0 discriminator 3
	movl	%eax, 32(%rsp)
	movl	%eax, 36(%rsp)
.LVL2083:
.LEHB24:
.LBE11659:
.LBE11658:
	.loc 1 940 0 discriminator 3
	call	_ZN2cv12_OutputArrayC1ERNS_3MatE
.LVL2084:
	.loc 1 940 0 is_stmt 0 discriminator 2
	leaq	240(%rsp), %rsi
.LVL2085:
	leaq	48(%rsp), %rdi
.LVL2086:
	call	_ZN2cv11_InputArrayC1ERKNS_3MatE
.LVL2087:
	.loc 1 940 0 discriminator 4
	movsd	8(%rsp), %xmm2
	leaq	32(%rsp), %rdx
	leaq	80(%rsp), %rsi
	leaq	48(%rsp), %rdi
.LVL2088:
	movl	$4, %ecx
	movapd	%xmm2, %xmm1
	movapd	%xmm2, %xmm0
	call	_ZN2cv12GaussianBlurERKNS_11_InputArrayERKNS_12_OutputArrayENS_5Size_IiEEddi
.LVL2089:
	.loc 1 941 0 is_stmt 1 discriminator 6
	call	omp_get_wtime
.LVL2090:
	movapd	%xmm0, %xmm7
	subsd	16(%rsp), %xmm7
	movsd	%xmm7, (%rsp)
.LVL2091:
	.loc 1 942 0 discriminator 6
	call	clock
.LVL2092:
	.loc 1 943 0 discriminator 6
	pxor	%xmm1, %xmm1
	subq	%rbx, %rax
.LVL2093:
.LBB11660:
.LBB11661:
	.loc 12 561 0 discriminator 6
	movl	$40, %edx
	movl	$.LC61, %esi
	movl	$_ZSt4cout, %edi
.LBE11661:
.LBE11660:
	.loc 1 943 0 discriminator 6
	cvtsi2ssq	%rax, %xmm1
	movss	%xmm1, 16(%rsp)
.LBB11663:
.LBB11662:
	.loc 12 561 0 discriminator 6
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.LVL2094:
.LBE11662:
.LBE11663:
.LBB11664:
.LBB11665:
	.loc 12 228 0
	pxor	%xmm0, %xmm0
	movl	$_ZSt4cout, %edi
	cvtss2sd	16(%rsp), %xmm0
	call	_ZNSo9_M_insertIdEERSoT_
.LVL2095:
.LBE11665:
.LBE11664:
	.loc 1 943 0
	movq	%rax, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.constprop.80
.LVL2096:
.LBB11666:
.LBB11667:
	.loc 12 561 0
	movl	$40, %edx
	movl	$.LC62, %esi
	movl	$_ZSt4cout, %edi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.LVL2097:
.LBE11667:
.LBE11666:
.LBB11668:
.LBB11669:
	.loc 12 221 0
	movsd	(%rsp), %xmm0
	movl	$_ZSt4cout, %edi
	call	_ZNSo9_M_insertIdEERSoT_
.LVL2098:
.LBE11669:
.LBE11668:
	.loc 1 944 0
	movl	$.LC59, %esi
	movq	%rax, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.LVL2099:
.LEHE24:
	.loc 1 945 0
	leaq	432(%rsp), %rsi
	leaq	80(%rsp), %rdi
.LBB11670:
.LBB11671:
.LBB11672:
.LBB11673:
	.loc 15 87 0
	movq	$0, 48(%rsp)
	movq	$0, 56(%rsp)
	movq	$0, 64(%rsp)
.LVL2100:
.LEHB25:
.LBE11673:
.LBE11672:
.LBE11671:
.LBE11670:
	.loc 1 945 0
	call	_ZN2cv11_InputArrayC1ERKNS_3MatE
.LVL2101:
	.loc 1 945 0 is_stmt 0 discriminator 2
	leaq	144(%rsp), %rdi
	movl	$.LC63, %esi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_.isra.64
.LVL2102:
.LEHE25:
	.loc 1 945 0 discriminator 4
	leaq	48(%rsp), %rdx
.LVL2103:
	leaq	80(%rsp), %rsi
	leaq	144(%rsp), %rdi
.LEHB26:
	call	_ZN2cv7imwriteERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKNS_11_InputArrayERKSt6vectorIiSaIiEE
.LVL2104:
.LEHE26:
	movq	144(%rsp), %rdi
.LVL2105:
.LBB11674:
.LBB11675:
.LBB11676:
	.loc 9 179 0 is_stmt 1
	leaq	160(%rsp), %rax
	cmpq	%rax, %rdi
	je	.L1041
.LVL2106:
.LBB11677:
.LBB11678:
.LBB11679:
.LBB11680:
	.loc 17 110 0
	call	_ZdlPv
.LVL2107:
.L1041:
.LBE11680:
.LBE11679:
.LBE11678:
.LBE11677:
.LBE11676:
.LBE11675:
.LBE11674:
.LBB11681:
.LBB11682:
.LBB11683:
	.loc 15 161 0
	movq	48(%rsp), %rdi
.LVL2108:
.LBB11684:
.LBB11685:
	.loc 15 177 0
	testq	%rdi, %rdi
	je	.L1042
.LVL2109:
.LBB11686:
.LBB11687:
.LBB11688:
	.loc 17 110 0
	call	_ZdlPv
.LVL2110:
.L1042:
.LBE11688:
.LBE11687:
.LBE11686:
.LBE11685:
.LBE11684:
.LBE11683:
.LBE11682:
.LBE11681:
	.loc 1 946 0
	leaq	336(%rsp), %rdx
	leaq	432(%rsp), %rsi
	leaq	736(%rsp), %rdi
.LBB11689:
.LBB11690:
.LBB11691:
.LBB11692:
	.loc 15 87 0
	movq	$0, 48(%rsp)
	movq	$0, 56(%rsp)
	movq	$0, 64(%rsp)
.LVL2111:
.LEHB27:
.LBE11692:
.LBE11691:
.LBE11690:
.LBE11689:
	.loc 1 946 0
	call	_ZN2cvmiERKNS_3MatES2_
.LVL2112:
.LEHE27:
	.loc 1 946 0 is_stmt 0 discriminator 2
	leaq	736(%rsp), %rsi
	leaq	80(%rsp), %rdi
.LEHB28:
	call	_ZN2cv11_InputArrayC1ERKNS_7MatExprE
.LVL2113:
	.loc 1 946 0 discriminator 4
	leaq	176(%rsp), %rdi
	movl	$.LC64, %esi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_.isra.64
.LVL2114:
.LEHE28:
	.loc 1 946 0 discriminator 6
	leaq	48(%rsp), %rdx
.LVL2115:
	leaq	80(%rsp), %rsi
	leaq	176(%rsp), %rdi
.LEHB29:
	call	_ZN2cv7imwriteERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKNS_11_InputArrayERKSt6vectorIiSaIiEE
.LVL2116:
.LEHE29:
	movq	176(%rsp), %rdi
.LVL2117:
.LBB11693:
.LBB11694:
.LBB11695:
	.loc 9 179 0 is_stmt 1
	leaq	192(%rsp), %rax
	cmpq	%rax, %rdi
	je	.L1043
.LVL2118:
.LBB11696:
.LBB11697:
.LBB11698:
.LBB11699:
	.loc 17 110 0
	call	_ZdlPv
.LVL2119:
.L1043:
.LBE11699:
.LBE11698:
.LBE11697:
.LBE11696:
.LBE11695:
.LBE11694:
.LBE11693:
	.loc 1 946 0
	leaq	736(%rsp), %rdi
	call	_ZN2cv7MatExprD1Ev
.LVL2120:
.LBB11700:
.LBB11701:
.LBB11702:
	.loc 15 161 0
	movq	48(%rsp), %rdi
.LVL2121:
.LBB11703:
.LBB11704:
	.loc 15 177 0
	testq	%rdi, %rdi
	je	.L1044
.LVL2122:
.LBB11705:
.LBB11706:
.LBB11707:
	.loc 17 110 0
	call	_ZdlPv
.LVL2123:
.L1044:
.LBE11707:
.LBE11706:
.LBE11705:
.LBE11704:
.LBE11703:
.LBE11702:
.LBE11701:
.LBE11700:
	.loc 1 948 0
	pxor	%xmm2, %xmm2
	leaq	1088(%rsp), %rdi
	movsd	8(%rsp), %xmm0
	movsd	.LC15(%rip), %xmm3
	cvtss2sd	24(%rsp), %xmm2
	movapd	%xmm0, %xmm1
	call	_ZN15BilateralFilterC1Edddd
.LVL2124:
.LBB11708:
.LBB11709:
	.loc 2 713 0
	movq	304(%rsp), %rax
.LBE11709:
.LBE11708:
.LBB11716:
.LBB11717:
	.loc 2 400 0
	movl	240(%rsp), %edx
.LBE11717:
.LBE11716:
	.loc 1 949 0
	leaq	80(%rsp), %rsi
	leaq	528(%rsp), %rdi
.LBB11719:
.LBB11714:
	.loc 2 713 0
	movl	(%rax), %ecx
.LBB11710:
.LBB11711:
	.loc 13 1861 0
	movl	4(%rax), %eax
.LBE11711:
.LBE11710:
.LBE11714:
.LBE11719:
.LBB11720:
.LBB11718:
	.loc 2 400 0
	andl	$4095, %edx
.LVL2125:
.LBE11718:
.LBE11720:
.LBB11721:
.LBB11715:
.LBB11713:
.LBB11712:
	.loc 13 1861 0
	movl	%eax, 80(%rsp)
	movl	%ecx, 84(%rsp)
.LVL2126:
.LEHB30:
.LBE11712:
.LBE11713:
.LBE11715:
.LBE11721:
	.loc 1 949 0
	call	_ZN2cv3MatC1ENS_5Size_IiEEi
.LVL2127:
.LEHE30:
	.loc 1 950 0 discriminator 3
	leaq	528(%rsp), %rdx
	leaq	240(%rsp), %rsi
.LVL2128:
	leaq	1088(%rsp), %rdi
.LEHB31:
	call	_ZN15BilateralFilter5applyERKN2cv3MatERS1_
.LVL2129:
.LEHE31:
	.loc 1 952 0
	leaq	528(%rsp), %rsi
	leaq	80(%rsp), %rdi
.LBB11722:
.LBB11723:
.LBB11724:
.LBB11725:
	.loc 15 87 0
	movq	$0, 48(%rsp)
	movq	$0, 56(%rsp)
	movq	$0, 64(%rsp)
.LVL2130:
.LEHB32:
.LBE11725:
.LBE11724:
.LBE11723:
.LBE11722:
	.loc 1 952 0
	call	_ZN2cv11_InputArrayC1ERKNS_3MatE
.LVL2131:
	.loc 1 952 0 is_stmt 0 discriminator 2
	leaq	208(%rsp), %rdi
	movl	$.LC65, %esi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_.isra.64
.LVL2132:
.LEHE32:
	.loc 1 952 0 discriminator 4
	leaq	48(%rsp), %rdx
.LVL2133:
	leaq	80(%rsp), %rsi
	leaq	208(%rsp), %rdi
.LEHB33:
	call	_ZN2cv7imwriteERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKNS_11_InputArrayERKSt6vectorIiSaIiEE
.LVL2134:
.LEHE33:
	movq	208(%rsp), %rdi
.LVL2135:
.LBB11726:
.LBB11727:
.LBB11728:
	.loc 9 179 0 is_stmt 1
	leaq	224(%rsp), %rax
	cmpq	%rax, %rdi
	je	.L1045
.LVL2136:
.LBB11729:
.LBB11730:
.LBB11731:
.LBB11732:
	.loc 17 110 0
	call	_ZdlPv
.LVL2137:
.L1045:
.LBE11732:
.LBE11731:
.LBE11730:
.LBE11729:
.LBE11728:
.LBE11727:
.LBE11726:
.LBB11733:
.LBB11734:
.LBB11735:
	.loc 15 161 0
	movq	48(%rsp), %rdi
.LVL2138:
.LBB11736:
.LBB11737:
	.loc 15 177 0
	testq	%rdi, %rdi
	je	.L1046
.LVL2139:
.LBB11738:
.LBB11739:
.LBB11740:
	.loc 17 110 0
	call	_ZdlPv
.LVL2140:
.L1046:
.LBE11740:
.LBE11739:
.LBE11738:
.LBE11737:
.LBE11736:
.LBE11735:
.LBE11734:
.LBE11733:
	.loc 1 949 0
	leaq	528(%rsp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL2141:
	.loc 1 937 0
	leaq	432(%rsp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL2142:
	.loc 1 922 0
	leaq	336(%rsp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL2143:
	.loc 1 919 0
	leaq	240(%rsp), %rdi
.LVL2144:
	call	_ZN2cv3MatD1Ev
.LVL2145:
	.loc 1 965 0
	xorl	%eax, %eax
.LVL2146:
.L1037:
	.loc 1 966 0
	movq	3160(%rsp), %rcx
	xorq	%fs:40, %rcx
	jne	.L1117
	addq	$3168, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
.LVL2147:
.L1116:
	.cfi_restore_state
.LBB11741:
.LBB11742:
	.loc 8 98 0
	movq	stdout(%rip), %rdi
.LVL2148:
	movl	$.LC53, %edx
	movl	$1, %esi
.LVL2149:
.LEHB34:
	call	__fprintf_chk
.LVL2150:
.LBE11742:
.LBE11741:
	.loc 1 912 0
	orl	$-1, %eax
	jmp	.L1037
.LVL2151:
.L1117:
	.loc 1 966 0
	call	__stack_chk_fail
.LVL2152:
.L1080:
	movq	%rax, %rbx
.LVL2153:
.L1061:
	.loc 1 946 0 discriminator 3
	leaq	736(%rsp), %rdi
	call	_ZN2cv7MatExprD1Ev
.LVL2154:
.L1062:
.LBB11743:
.LBB11744:
.LBB11745:
	.loc 15 161 0 discriminator 1
	movq	48(%rsp), %rdi
.LVL2155:
.LBB11746:
.LBB11747:
	.loc 15 177 0 discriminator 1
	testq	%rdi, %rdi
	je	.L1058
.LVL2156:
.LBB11748:
.LBB11749:
.LBB11750:
	.loc 17 110 0
	call	_ZdlPv
.LVL2157:
.L1058:
.LBE11750:
.LBE11749:
.LBE11748:
.LBE11747:
.LBE11746:
.LBE11745:
.LBE11744:
.LBE11743:
	.loc 1 937 0
	leaq	432(%rsp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL2158:
.L1053:
	.loc 1 922 0
	leaq	336(%rsp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL2159:
.L1069:
	.loc 1 919 0
	leaq	240(%rsp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL2160:
.L1112:
	movq	%rbx, %rdi
	call	_Unwind_Resume
.LVL2161:
.LEHE34:
.L1079:
.L1114:
	movq	%rax, %rbx
.LVL2162:
	jmp	.L1062
.LVL2163:
.L1078:
	movq	144(%rsp), %rdi
.LBB11751:
.LBB11752:
.LBB11753:
	.loc 9 179 0
	leaq	160(%rsp), %rdx
	movq	%rax, %rbx
.LVL2164:
	cmpq	%rdx, %rdi
	je	.L1062
.LVL2165:
.LBB11754:
.LBB11755:
.LBB11756:
.LBB11757:
	.loc 17 110 0
	call	_ZdlPv
.LVL2166:
	jmp	.L1062
.LVL2167:
.L1083:
	movq	%rax, %rbx
.LVL2168:
.L1066:
.LBE11757:
.LBE11756:
.LBE11755:
.LBE11754:
.LBE11753:
.LBE11752:
.LBE11751:
.LBB11758:
.LBB11759:
.LBB11760:
	.loc 15 161 0 discriminator 1
	movq	48(%rsp), %rdi
.LVL2169:
.LBB11761:
.LBB11762:
	.loc 15 177 0 discriminator 1
	testq	%rdi, %rdi
	je	.L1068
.LVL2170:
.LBB11763:
.LBB11764:
.LBB11765:
	.loc 17 110 0
	call	_ZdlPv
.LVL2171:
.L1068:
.LBE11765:
.LBE11764:
.LBE11763:
.LBE11762:
.LBE11761:
.LBE11760:
.LBE11759:
.LBE11758:
	.loc 1 949 0
	leaq	528(%rsp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL2172:
	jmp	.L1058
.LVL2173:
.L1082:
	movq	%rax, %rbx
.LVL2174:
	jmp	.L1068
.LVL2175:
.L1084:
	movq	208(%rsp), %rdi
.LBB11766:
.LBB11767:
.LBB11768:
	.loc 9 179 0
	leaq	224(%rsp), %rdx
	movq	%rax, %rbx
.LVL2176:
	cmpq	%rdx, %rdi
	je	.L1066
.LVL2177:
.LBB11769:
.LBB11770:
.LBB11771:
.LBB11772:
	.loc 17 110 0
	call	_ZdlPv
.LVL2178:
	jmp	.L1066
.LVL2179:
.L1081:
	movq	176(%rsp), %rdi
.LBE11772:
.LBE11771:
.LBE11770:
.LBE11769:
.LBE11768:
.LBE11767:
.LBE11766:
.LBB11773:
.LBB11774:
.LBB11775:
	.loc 9 179 0
	leaq	192(%rsp), %rdx
	movq	%rax, %rbx
.LVL2180:
	cmpq	%rdx, %rdi
	je	.L1061
.LVL2181:
.LBB11776:
.LBB11777:
.LBB11778:
.LBB11779:
	.loc 17 110 0
	call	_ZdlPv
.LVL2182:
	jmp	.L1061
.LVL2183:
.L1072:
	movq	%rax, %rbx
	jmp	.L1069
.LVL2184:
.L1074:
	movq	%rax, %rbx
.LVL2185:
.L1051:
.LBE11779:
.LBE11778:
.LBE11777:
.LBE11776:
.LBE11775:
.LBE11774:
.LBE11773:
.LBB11780:
.LBB11781:
.LBB11782:
	.loc 15 161 0 discriminator 1
	movq	48(%rsp), %rdi
.LVL2186:
.LBB11783:
.LBB11784:
	.loc 15 177 0 discriminator 1
	testq	%rdi, %rdi
	je	.L1053
.LVL2187:
.LBB11785:
.LBB11786:
.LBB11787:
	.loc 17 110 0
	call	_ZdlPv
.LVL2188:
	jmp	.L1053
.LVL2189:
.L1073:
	movq	%rax, %rbx
	jmp	.L1053
.LVL2190:
.L1071:
	movq	1088(%rsp), %rdi
.LBE11787:
.LBE11786:
.LBE11785:
.LBE11784:
.LBE11783:
.LBE11782:
.LBE11781:
.LBE11780:
.LBB11788:
.LBB11789:
.LBB11790:
	.loc 9 179 0
	leaq	1104(%rsp), %rdx
	movq	%rax, %rbx
.LVL2191:
	cmpq	%rdx, %rdi
	je	.L1112
.LVL2192:
.LBB11791:
.LBB11792:
.LBB11793:
.LBB11794:
	.loc 17 110 0
	call	_ZdlPv
.LVL2193:
	jmp	.L1112
.LVL2194:
.L1077:
	jmp	.L1114
.LVL2195:
.L1076:
	movq	%rax, %rbx
.LVL2196:
	jmp	.L1058
.LVL2197:
.L1075:
	movq	112(%rsp), %rdi
.LBE11794:
.LBE11793:
.LBE11792:
.LBE11791:
.LBE11790:
.LBE11789:
.LBE11788:
.LBB11795:
.LBB11796:
.LBB11797:
	.loc 9 179 0
	leaq	128(%rsp), %rdx
	movq	%rax, %rbx
.LVL2198:
	cmpq	%rdx, %rdi
	je	.L1051
.LVL2199:
.LBB11798:
.LBB11799:
.LBB11800:
.LBB11801:
	.loc 17 110 0
	call	_ZdlPv
.LVL2200:
	jmp	.L1051
.LBE11801:
.LBE11800:
.LBE11799:
.LBE11798:
.LBE11797:
.LBE11796:
.LBE11795:
	.cfi_endproc
.LFE11299:
	.section	.gcc_except_table
.LLSDA11299:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE11299-.LLSDACSB11299
.LLSDACSB11299:
	.uleb128 .LEHB17-.LFB11299
	.uleb128 .LEHE17-.LEHB17
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB18-.LFB11299
	.uleb128 .LEHE18-.LEHB18
	.uleb128 .L1071-.LFB11299
	.uleb128 0
	.uleb128 .LEHB19-.LFB11299
	.uleb128 .LEHE19-.LEHB19
	.uleb128 .L1072-.LFB11299
	.uleb128 0
	.uleb128 .LEHB20-.LFB11299
	.uleb128 .LEHE20-.LEHB20
	.uleb128 .L1073-.LFB11299
	.uleb128 0
	.uleb128 .LEHB21-.LFB11299
	.uleb128 .LEHE21-.LEHB21
	.uleb128 .L1074-.LFB11299
	.uleb128 0
	.uleb128 .LEHB22-.LFB11299
	.uleb128 .LEHE22-.LEHB22
	.uleb128 .L1075-.LFB11299
	.uleb128 0
	.uleb128 .LEHB23-.LFB11299
	.uleb128 .LEHE23-.LEHB23
	.uleb128 .L1073-.LFB11299
	.uleb128 0
	.uleb128 .LEHB24-.LFB11299
	.uleb128 .LEHE24-.LEHB24
	.uleb128 .L1076-.LFB11299
	.uleb128 0
	.uleb128 .LEHB25-.LFB11299
	.uleb128 .LEHE25-.LEHB25
	.uleb128 .L1077-.LFB11299
	.uleb128 0
	.uleb128 .LEHB26-.LFB11299
	.uleb128 .LEHE26-.LEHB26
	.uleb128 .L1078-.LFB11299
	.uleb128 0
	.uleb128 .LEHB27-.LFB11299
	.uleb128 .LEHE27-.LEHB27
	.uleb128 .L1079-.LFB11299
	.uleb128 0
	.uleb128 .LEHB28-.LFB11299
	.uleb128 .LEHE28-.LEHB28
	.uleb128 .L1080-.LFB11299
	.uleb128 0
	.uleb128 .LEHB29-.LFB11299
	.uleb128 .LEHE29-.LEHB29
	.uleb128 .L1081-.LFB11299
	.uleb128 0
	.uleb128 .LEHB30-.LFB11299
	.uleb128 .LEHE30-.LEHB30
	.uleb128 .L1076-.LFB11299
	.uleb128 0
	.uleb128 .LEHB31-.LFB11299
	.uleb128 .LEHE31-.LEHB31
	.uleb128 .L1082-.LFB11299
	.uleb128 0
	.uleb128 .LEHB32-.LFB11299
	.uleb128 .LEHE32-.LEHB32
	.uleb128 .L1083-.LFB11299
	.uleb128 0
	.uleb128 .LEHB33-.LFB11299
	.uleb128 .LEHE33-.LEHB33
	.uleb128 .L1084-.LFB11299
	.uleb128 0
	.uleb128 .LEHB34-.LFB11299
	.uleb128 .LEHE34-.LEHB34
	.uleb128 0
	.uleb128 0
.LLSDACSE11299:
	.section	.text.startup
	.size	main, .-main
	.section	.text.unlikely
.LCOLDE66:
	.section	.text.startup
.LHOTE66:
	.section	.rodata.str1.1
.LC68:
	.string	"kx % 2 == 1 && ky % 2 == 1"
.LC69:
	.string	"sigmaX > 0 && sigmaY > 0"
	.section	.rodata.str1.8
	.align 8
.LC70:
	.string	"src.rows == dst.rows && src.cols == dst.cols"
	.section	.rodata.str1.1
.LC71:
	.string	"w % 16 == 0"
	.section	.text.unlikely._Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LCOLDB72:
	.section	.text._Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LHOTB72:
	.p2align 4,,15
	.weak	_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd
	.type	_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd, @function
_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd:
.LFB11596:
	.loc 1 40 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA11596
.LVL2201:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movapd	%xmm0, %xmm3
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$680, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	.loc 1 40 0
	movq	%rsi, -672(%rbp)
	movsd	.LC67(%rip), %xmm2
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
.LVL2202:
	mulsd	%xmm2, %xmm3
	mulsd	%xmm1, %xmm2
	cvttsd2si	%xmm3, %eax
	leal	1(%rax,%rax), %ecx
	movl	(%rdx), %eax
	cmpl	%eax, %ecx
	cmovg	%eax, %ecx
.LVL2203:
	cvttsd2si	%xmm2, %eax
	.loc 1 45 0
	movl	%ecx, -600(%rbp)
.LVL2204:
	leal	1(%rax,%rax), %esi
.LVL2205:
	movl	4(%rdx), %eax
	cmpl	%eax, %esi
	cmovle	%esi, %eax
.LVL2206:
	.loc 1 47 0
	movl	%ecx, %esi
	shrl	$31, %esi
	.loc 1 46 0
	movl	%eax, -596(%rbp)
	.loc 1 47 0
	leal	(%rcx,%rsi), %edx
.LVL2207:
	andl	$1, %edx
	subl	%esi, %edx
	cmpl	$1, %edx
	jne	.L1119
	.loc 1 47 0 is_stmt 0 discriminator 2
	movl	%eax, %esi
	shrl	$31, %esi
	leal	(%rax,%rsi), %edx
	andl	$1, %edx
	subl	%esi, %edx
	cmpl	$1, %edx
	jne	.L1119
	.loc 1 49 0 is_stmt 1
	movl	%eax, %edx
	.loc 1 48 0
	movl	%ecx, %ebx
	.loc 1 49 0
	shrl	$31, %edx
	.loc 1 48 0
	shrl	$31, %ebx
	.loc 1 49 0
	addl	%edx, %eax
	.loc 1 48 0
	addl	%ecx, %ebx
	.loc 1 49 0
	sarl	%eax
	.loc 1 48 0
	sarl	%ebx
.LVL2208:
	.loc 1 51 0
	leal	1(%rax), %ecx
	.loc 1 49 0
	movl	%eax, -676(%rbp)
.LVL2209:
	.loc 1 50 0
	leal	1(%rbx), %edx
	.loc 1 54 0
	pxor	%xmm7, %xmm7
	.loc 1 52 0
	movslq	%ecx, %rax
.LVL2210:
	.loc 1 50 0
	movl	%edx, -600(%rbp)
	.loc 1 51 0
	movl	%ecx, -596(%rbp)
.LVL2211:
	.loc 1 52 0
	leaq	18(,%rax,4), %rax
	andq	$-16, %rax
	subq	%rax, %rsp
	leaq	3(%rsp), %r14
	shrq	$2, %r14
	leaq	0(,%r14,4), %rax
	movq	%rax, -656(%rbp)
.LVL2212:
	.loc 1 53 0
	movslq	%edx, %rax
.LVL2213:
	leaq	18(,%rax,4), %rax
	andq	$-16, %rax
	subq	%rax, %rsp
	leaq	3(%rsp), %r12
	shrq	$2, %r12
	.loc 1 54 0
	ucomisd	%xmm7, %xmm0
	.loc 1 53 0
	leaq	0(,%r12,4), %rax
	movq	%rax, -664(%rbp)
.LVL2214:
	.loc 1 54 0
	jbe	.L1121
	.loc 1 54 0 is_stmt 0 discriminator 2
	ucomisd	%xmm7, %xmm1
	jbe	.L1121
	.loc 1 55 0 is_stmt 1
	movsd	.LC38(%rip), %xmm3
	movapd	%xmm0, %xmm4
	movsd	.LC15(%rip), %xmm2
.LBB11896:
	.loc 1 60 0
	testl	%ecx, %ecx
.LBE11896:
	.loc 1 55 0
	mulsd	%xmm3, %xmm4
	movq	%rdi, %r13
	.loc 1 56 0
	mulsd	%xmm1, %xmm3
	movapd	%xmm2, %xmm6
	.loc 1 55 0
	movapd	%xmm2, %xmm5
	divsd	%xmm4, %xmm5
	.loc 1 56 0
	divsd	%xmm3, %xmm6
	.loc 1 57 0
	movapd	%xmm0, %xmm3
	addsd	%xmm0, %xmm3
	.loc 1 55 0
	movsd	%xmm5, -624(%rbp)
.LVL2215:
	.loc 1 57 0
	mulsd	%xmm3, %xmm0
.LVL2216:
	movapd	%xmm2, %xmm3
	.loc 1 56 0
	movsd	%xmm6, -632(%rbp)
.LVL2217:
	.loc 1 57 0
	divsd	%xmm0, %xmm3
	.loc 1 58 0
	movapd	%xmm1, %xmm0
	addsd	%xmm1, %xmm0
	mulsd	%xmm0, %xmm1
.LVL2218:
	.loc 1 57 0
	movsd	%xmm3, -640(%rbp)
.LVL2219:
	.loc 1 58 0
	divsd	%xmm1, %xmm2
	movsd	%xmm2, -648(%rbp)
.LVL2220:
.LBB11897:
	.loc 1 60 0
	jle	.L1124
	xorl	%r15d, %r15d
	movl	%ebx, -680(%rbp)
	pxor	%xmm1, %xmm1
	movl	%r15d, %ebx
.LVL2221:
	movq	-656(%rbp), %r15
	jmp	.L1130
.LVL2222:
	.p2align 4,,10
	.p2align 3
.L1288:
	.loc 1 62 0 discriminator 1
	addsd	%xmm0, %xmm0
	.loc 1 60 0 discriminator 1
	movl	-596(%rbp), %eax
	addl	$1, %ebx
.LVL2223:
	cmpl	%ebx, %eax
	.loc 1 62 0 discriminator 1
	addsd	%xmm0, %xmm1
.LVL2224:
	.loc 1 60 0 discriminator 1
	jle	.L1287
.LVL2225:
.L1130:
	.loc 1 61 0
	movl	%ebx, %eax
	pxor	%xmm0, %xmm0
	negl	%eax
	movsd	%xmm1, -616(%rbp)
.LVL2226:
	imull	%ebx, %eax
	cvtsi2sd	%eax, %xmm0
	mulsd	-648(%rbp), %xmm0
	call	exp
.LVL2227:
	mulsd	-632(%rbp), %xmm0
	movslq	%ebx, %rax
	.loc 1 62 0
	testl	%ebx, %ebx
	movsd	-616(%rbp), %xmm1
	.loc 1 61 0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, (%r15,%rax,4)
	.loc 1 62 0
	cvtss2sd	%xmm0, %xmm0
	jg	.L1288
.LVL2228:
	.loc 1 60 0
	movl	-596(%rbp), %eax
	addl	$1, %ebx
.LVL2229:
	.loc 1 63 0
	addsd	%xmm0, %xmm1
.LVL2230:
	.loc 1 60 0
	cmpl	%ebx, %eax
	jg	.L1130
.L1287:
.LBE11897:
.LBB11898:
	.loc 1 65 0
	testl	%eax, %eax
	movl	-680(%rbp), %ebx
.LVL2231:
	jle	.L1137
	movq	-656(%rbp), %rdx
	andl	$15, %edx
	shrq	$2, %rdx
	negq	%rdx
	andl	$3, %edx
	cmpl	%eax, %edx
	cmova	%eax, %edx
	cmpl	$4, %eax
	jg	.L1289
	movl	%eax, %edx
.L1134:
	pxor	%xmm0, %xmm0
	cmpl	$1, %edx
	movl	$1, %ecx
	pxor	%xmm4, %xmm4
	cvtss2sd	0(,%r14,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm4
	movss	%xmm4, 0(,%r14,4)
.LVL2232:
	je	.L1136
	pxor	%xmm0, %xmm0
	cmpl	$2, %edx
	movl	$2, %ecx
	pxor	%xmm5, %xmm5
	cvtss2sd	4(,%r14,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm5
	movss	%xmm5, 4(,%r14,4)
.LVL2233:
	je	.L1136
	pxor	%xmm0, %xmm0
	cmpl	$3, %edx
	movl	$3, %ecx
	pxor	%xmm6, %xmm6
	cvtss2sd	8(,%r14,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm6
	movss	%xmm6, 8(,%r14,4)
.LVL2234:
	je	.L1136
	pxor	%xmm0, %xmm0
	movl	$4, %ecx
	pxor	%xmm7, %xmm7
	cvtss2sd	12(,%r14,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm7
	movss	%xmm7, 12(,%r14,4)
.LVL2235:
.L1136:
	cmpl	%edx, %eax
	je	.L1137
.L1135:
	movl	%eax, %r8d
	leal	-1(%rax), %edi
	movl	%edx, %r10d
	subl	%edx, %r8d
	leal	-4(%r8), %esi
	subl	%edx, %edi
	shrl	$2, %esi
	addl	$1, %esi
	cmpl	$2, %edi
	leal	0(,%rsi,4), %r9d
	jbe	.L1138
	movq	-656(%rbp), %rdi
	movddup	%xmm1, %xmm3
	xorl	%edx, %edx
	leaq	(%rdi,%r10,4), %r10
	xorl	%edi, %edi
.L1140:
	.loc 1 65 0 is_stmt 0 discriminator 2
	movaps	(%r10,%rdx), %xmm2
	addl	$1, %edi
	movhps	%xmm2, -704(%rbp)
	cvtps2pd	%xmm2, %xmm0
	cvtps2pd	-704(%rbp), %xmm2
	divpd	%xmm3, %xmm0
	divpd	%xmm3, %xmm2
	cvtpd2ps	%xmm0, %xmm0
	cvtpd2ps	%xmm2, %xmm2
	movlhps	%xmm2, %xmm0
	movaps	%xmm0, (%r10,%rdx)
	addq	$16, %rdx
	cmpl	%edi, %esi
	ja	.L1140
	addl	%r9d, %ecx
	cmpl	%r8d, %r9d
	je	.L1137
.L1138:
.LVL2236:
	movq	-656(%rbp), %rsi
	movslq	%ecx, %rdx
	.loc 1 65 0
	pxor	%xmm0, %xmm0
	pxor	%xmm4, %xmm4
	leaq	(%rsi,%rdx,4), %rdx
	cvtss2sd	(%rdx), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm4
	movss	%xmm4, (%rdx)
	leal	1(%rcx), %edx
.LVL2237:
	cmpl	%edx, %eax
	jle	.L1137
	movslq	%edx, %rdx
	pxor	%xmm0, %xmm0
	leaq	(%rsi,%rdx,4), %rdx
.LVL2238:
	pxor	%xmm5, %xmm5
	addl	$2, %ecx
.LVL2239:
	cvtss2sd	(%rdx), %xmm0
	cmpl	%ecx, %eax
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm5
	movss	%xmm5, (%rdx)
.LVL2240:
	jle	.L1137
	movslq	%ecx, %rcx
	pxor	%xmm0, %xmm0
	leaq	(%rsi,%rcx,4), %rax
	pxor	%xmm7, %xmm7
	cvtss2sd	(%rax), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm7
	movss	%xmm7, (%rax)
.L1137:
	movl	-600(%rbp), %edx
.LVL2241:
.L1124:
.LBE11898:
.LBB11899:
	.loc 1 67 0 is_stmt 1 discriminator 1
	xorl	%r14d, %r14d
	testl	%edx, %edx
	movq	-664(%rbp), %r15
	pxor	%xmm1, %xmm1
	jg	.L1257
	jmp	.L1143
.LVL2242:
	.p2align 4,,10
	.p2align 3
.L1291:
	.loc 1 69 0 discriminator 1
	addsd	%xmm0, %xmm0
	.loc 1 67 0 discriminator 1
	movl	-600(%rbp), %eax
	addl	$1, %r14d
.LVL2243:
	cmpl	%r14d, %eax
	.loc 1 69 0 discriminator 1
	addsd	%xmm0, %xmm1
.LVL2244:
	.loc 1 67 0 discriminator 1
	jle	.L1290
.LVL2245:
.L1257:
	.loc 1 68 0
	movl	%r14d, %eax
	pxor	%xmm0, %xmm0
	negl	%eax
	movsd	%xmm1, -616(%rbp)
.LVL2246:
	imull	%r14d, %eax
	cvtsi2sd	%eax, %xmm0
	mulsd	-640(%rbp), %xmm0
	call	exp
.LVL2247:
	mulsd	-624(%rbp), %xmm0
	movslq	%r14d, %rax
	.loc 1 69 0
	testl	%r14d, %r14d
	movsd	-616(%rbp), %xmm1
	.loc 1 68 0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, (%r15,%rax,4)
	.loc 1 69 0
	cvtss2sd	%xmm0, %xmm0
	jg	.L1291
.LVL2248:
	.loc 1 67 0
	movl	-600(%rbp), %eax
	addl	$1, %r14d
.LVL2249:
	.loc 1 70 0
	addsd	%xmm0, %xmm1
.LVL2250:
	.loc 1 67 0
	cmpl	%r14d, %eax
	jg	.L1257
.L1290:
.LVL2251:
.LBE11899:
.LBB11900:
	.loc 1 72 0 discriminator 3
	testl	%eax, %eax
	jle	.L1143
	movq	-664(%rbp), %rdx
	andl	$15, %edx
	shrq	$2, %rdx
	negq	%rdx
	andl	$3, %edx
	cmpl	%eax, %edx
	cmova	%eax, %edx
	cmpl	$4, %eax
	jg	.L1292
	.loc 1 72 0 is_stmt 0
	movl	%eax, %edx
.L1149:
	pxor	%xmm0, %xmm0
	cmpl	$1, %edx
	movl	$1, %ecx
	pxor	%xmm4, %xmm4
	cvtss2sd	0(,%r12,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm4
	movss	%xmm4, 0(,%r12,4)
.LVL2252:
	je	.L1151
	pxor	%xmm0, %xmm0
	cmpl	$2, %edx
	movl	$2, %ecx
	pxor	%xmm5, %xmm5
	cvtss2sd	4(,%r12,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm5
	movss	%xmm5, 4(,%r12,4)
.LVL2253:
	je	.L1151
	pxor	%xmm0, %xmm0
	cmpl	$3, %edx
	movl	$3, %ecx
	pxor	%xmm6, %xmm6
	cvtss2sd	8(,%r12,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm6
	movss	%xmm6, 8(,%r12,4)
.LVL2254:
	je	.L1151
	pxor	%xmm0, %xmm0
	movl	$4, %ecx
	pxor	%xmm3, %xmm3
	cvtss2sd	12(,%r12,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm3
	movss	%xmm3, 12(,%r12,4)
.LVL2255:
.L1151:
	cmpl	%edx, %eax
	je	.L1143
.L1150:
	movl	%eax, %r8d
	leal	-1(%rax), %edi
	movl	%edx, %r10d
	subl	%edx, %r8d
	leal	-4(%r8), %esi
	subl	%edx, %edi
	shrl	$2, %esi
	addl	$1, %esi
	cmpl	$2, %edi
	leal	0(,%rsi,4), %r9d
	jbe	.L1153
	movq	-664(%rbp), %rdi
	movddup	%xmm1, %xmm3
	xorl	%edx, %edx
	leaq	(%rdi,%r10,4), %r10
	xorl	%edi, %edi
.L1155:
	.loc 1 72 0 discriminator 2
	movaps	(%r10,%rdx), %xmm2
	addl	$1, %edi
	movhps	%xmm2, -720(%rbp)
	cvtps2pd	%xmm2, %xmm0
	cvtps2pd	-720(%rbp), %xmm2
	divpd	%xmm3, %xmm0
	divpd	%xmm3, %xmm2
	cvtpd2ps	%xmm0, %xmm0
	cvtpd2ps	%xmm2, %xmm2
	movlhps	%xmm2, %xmm0
	movaps	%xmm0, (%r10,%rdx)
	addq	$16, %rdx
	cmpl	%esi, %edi
	jb	.L1155
	addl	%r9d, %ecx
	cmpl	%r8d, %r9d
	je	.L1143
.L1153:
.LVL2256:
	movq	-664(%rbp), %rsi
	movslq	%ecx, %rdx
	.loc 1 72 0
	pxor	%xmm0, %xmm0
	pxor	%xmm3, %xmm3
	leaq	(%rsi,%rdx,4), %rdx
	cvtss2sd	(%rdx), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm3
	movss	%xmm3, (%rdx)
	leal	1(%rcx), %edx
.LVL2257:
	cmpl	%edx, %eax
	jle	.L1143
	movslq	%edx, %rdx
	pxor	%xmm0, %xmm0
	leaq	(%rsi,%rdx,4), %rdx
.LVL2258:
	pxor	%xmm7, %xmm7
	addl	$2, %ecx
.LVL2259:
	cvtss2sd	(%rdx), %xmm0
	cmpl	%ecx, %eax
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm7
	movss	%xmm7, (%rdx)
.LVL2260:
	jle	.L1143
	movslq	%ecx, %rcx
	pxor	%xmm0, %xmm0
	leaq	(%rsi,%rcx,4), %rax
	pxor	%xmm6, %xmm6
	cvtss2sd	(%rax), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm6
	movss	%xmm6, (%rax)
.LVL2261:
.L1143:
.LBE11900:
	.loc 1 75 0 is_stmt 1
	movl	12(%r13), %eax
	.loc 1 74 0
	movl	8(%r13), %r12d
.LVL2262:
	.loc 1 78 0
	testb	$15, %al
	.loc 1 75 0
	movl	%eax, -624(%rbp)
.LVL2263:
	.loc 1 78 0
	je	.L1279
	movl	%eax, %ecx
	.loc 1 78 0 is_stmt 0 discriminator 1
	addl	$15, %eax
.LVL2264:
	testl	%ecx, %ecx
	cmovns	%ecx, %eax
	andl	$-16, %eax
	addl	$16, %eax
.LVL2265:
.L1279:
	movl	%eax, %ecx
.LVL2266:
	movl	%eax, -616(%rbp)
.LVL2267:
	.loc 1 80 0 is_stmt 1 discriminator 1
	leal	(%rbx,%rbx), %eax
	leal	(%rcx,%rax), %edx
.LVL2268:
	.loc 1 81 0 discriminator 1
	addl	-624(%rbp), %eax
	movl	%eax, -640(%rbp)
.LVL2269:
	.loc 1 82 0 discriminator 1
	movl	-676(%rbp), %eax
.LVL2270:
	leal	(%r12,%rax,2), %eax
	movl	%eax, -632(%rbp)
.LVL2271:
	.loc 1 83 0 discriminator 1
	movq	-672(%rbp), %rax
.LVL2272:
	movq	8(%rax), %rax
	cmpq	%rax, 8(%r13)
	jne	.L1293
.LVL2273:
.LBB11901:
.LBB11902:
.LBB11903:
	.loc 2 709 0
	leaq	-448(%rbp), %rax
.LVL2274:
.LBE11903:
.LBE11902:
.LBB11906:
.LBB11907:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE11907:
.LBE11906:
.LBB11911:
.LBB11912:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm3
.LBE11912:
.LBE11911:
.LBE11901:
.LBB11926:
.LBB11927:
.LBB11928:
	.loc 2 353 0
	leaq	-256(%rbp), %rdi
.LBE11928:
.LBE11927:
.LBE11926:
.LBB11953:
.LBB11917:
.LBB11904:
	.loc 2 709 0
	addq	$8, %rax
.LVL2275:
.LBE11904:
.LBE11917:
.LBB11918:
.LBB11913:
	.loc 2 62 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
.LBE11913:
.LBE11918:
.LBB11919:
.LBB11905:
	.loc 2 709 0
	movq	%rax, -384(%rbp)
.LVL2276:
.LBE11905:
.LBE11919:
.LBB11920:
.LBB11908:
	.loc 2 738 0
	leaq	-448(%rbp), %rax
.LVL2277:
.LBE11908:
.LBE11920:
.LBB11921:
.LBB11914:
	.loc 2 62 0
	movq	$0, -416(%rbp)
.LBE11914:
.LBE11921:
.LBB11922:
.LBB11909:
	.loc 2 738 0
	movaps	%xmm0, -368(%rbp)
.LVL2278:
	addq	$80, %rax
.LVL2279:
.LBE11909:
.LBE11922:
.LBB11923:
.LBB11915:
	.loc 2 62 0
	movq	$0, -432(%rbp)
	.loc 2 63 0
	movq	$0, -424(%rbp)
	.loc 2 60 0
	movaps	%xmm3, -448(%rbp)
.LBE11915:
.LBE11923:
.LBE11953:
.LBB11954:
.LBB11932:
.LBB11929:
	.loc 2 353 0
	movl	$16, %ecx
	movl	$2, %esi
.LBE11929:
.LBE11932:
.LBE11954:
.LBB11955:
.LBB11956:
.LBB11957:
	.loc 2 738 0
	movaps	%xmm0, -272(%rbp)
.LBE11957:
.LBE11956:
.LBB11961:
.LBB11962:
	.loc 2 60 0
	movaps	%xmm3, -352(%rbp)
.LBE11962:
.LBE11961:
.LBE11955:
.LBB11978:
.LBB11933:
.LBB11934:
	.loc 2 738 0
	movaps	%xmm0, -176(%rbp)
.LBE11934:
.LBE11933:
.LBB11936:
.LBB11937:
	.loc 2 60 0
	movaps	%xmm3, -256(%rbp)
.LBE11937:
.LBE11936:
.LBE11978:
.LBB11979:
.LBB11924:
.LBB11910:
	.loc 2 738 0
	movq	%rax, -376(%rbp)
.LBE11910:
.LBE11924:
.LBE11979:
.LBB11980:
.LBB11966:
.LBB11967:
	.loc 2 709 0
	leaq	-352(%rbp), %rax
.LVL2280:
.LBE11967:
.LBE11966:
.LBE11980:
.LBB11981:
.LBB11925:
.LBB11916:
	.loc 2 64 0
	movq	$0, -392(%rbp)
.LVL2281:
.LBE11916:
.LBE11925:
.LBE11981:
.LBB11982:
.LBB11970:
.LBB11963:
	.loc 2 62 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
.LBE11963:
.LBE11970:
.LBB11971:
.LBB11968:
	.loc 2 709 0
	addq	$8, %rax
.LVL2282:
.LBE11968:
.LBE11971:
.LBB11972:
.LBB11964:
	.loc 2 62 0
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
.LBE11964:
.LBE11972:
.LBB11973:
.LBB11969:
	.loc 2 709 0
	movq	%rax, -288(%rbp)
.LVL2283:
.LBE11969:
.LBE11973:
.LBB11974:
.LBB11958:
	.loc 2 738 0
	leaq	-352(%rbp), %rax
.LVL2284:
.LBE11958:
.LBE11974:
.LBB11975:
.LBB11965:
	.loc 2 63 0
	movq	$0, -328(%rbp)
	.loc 2 64 0
	movq	$0, -296(%rbp)
.LVL2285:
.LBE11965:
.LBE11975:
.LBE11982:
.LBB11983:
.LBB11941:
.LBB11938:
	.loc 2 62 0
	movq	$0, -208(%rbp)
.LBE11938:
.LBE11941:
.LBE11983:
.LBB11984:
.LBB11976:
.LBB11959:
	.loc 2 738 0
	addq	$80, %rax
.LBE11959:
.LBE11976:
.LBE11984:
.LBB11985:
.LBB11942:
.LBB11939:
	.loc 2 62 0
	movq	$0, -216(%rbp)
	movq	$0, -224(%rbp)
.LBE11939:
.LBE11942:
.LBE11985:
.LBB11986:
.LBB11977:
.LBB11960:
	.loc 2 738 0
	movq	%rax, -280(%rbp)
.LBE11960:
.LBE11977:
.LBE11986:
.LBB11987:
.LBB11943:
.LBB11944:
	.loc 2 709 0
	leaq	-256(%rbp), %rax
.LBE11944:
.LBE11943:
.LBB11947:
.LBB11940:
	.loc 2 62 0
	movq	$0, -240(%rbp)
	.loc 2 63 0
	movq	$0, -232(%rbp)
	.loc 2 64 0
	movq	$0, -200(%rbp)
.LBE11940:
.LBE11947:
.LBB11948:
.LBB11945:
	.loc 2 709 0
	addq	$8, %rax
.LBE11945:
.LBE11948:
.LBB11949:
.LBB11930:
	.loc 2 352 0
	movl	%r12d, -480(%rbp)
.LBE11930:
.LBE11949:
.LBB11950:
.LBB11946:
	.loc 2 709 0
	movq	%rax, -192(%rbp)
.LVL2286:
.LBE11946:
.LBE11950:
.LBB11951:
.LBB11935:
	.loc 2 738 0
	leaq	-256(%rbp), %rax
	addq	$80, %rax
	movq	%rax, -184(%rbp)
.LBE11935:
.LBE11951:
.LBB11952:
.LBB11931:
	.loc 2 352 0
	movl	%edx, -476(%rbp)
	.loc 2 353 0
	leaq	-480(%rbp), %rdx
.LVL2287:
.LEHB35:
	call	_ZN2cv3Mat6createEiPKii
.LVL2288:
.LEHE35:
.LBE11931:
.LBE11952:
.LBE11987:
.LBB11988:
.LBB11989:
	.loc 2 285 0
	movq	-232(%rbp), %rax
	testq	%rax, %rax
	je	.L1159
	.loc 2 286 0
	lock addl	$1, (%rax)
.L1159:
.LVL2289:
.LBB11990:
.LBB11991:
	.loc 2 366 0
	movq	-424(%rbp), %rax
	testq	%rax, %rax
	je	.L1213
	lock subl	$1, (%rax)
	je	.L1294
.L1213:
.LBB11992:
	.loc 2 369 0
	movl	-444(%rbp), %edx
.LBE11992:
	.loc 2 368 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
.LVL2290:
.LBB11993:
	.loc 2 369 0
	testl	%edx, %edx
	jle	.L1295
	movq	-384(%rbp), %rdx
	xorl	%eax, %eax
.LVL2291:
	.p2align 4,,10
	.p2align 3
.L1162:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-444(%rbp), %ecx
	addl	$1, %eax
.LVL2292:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L1162
.LBE11993:
.LBE11991:
.LBE11990:
	.loc 2 288 0
	movl	-256(%rbp), %eax
.LVL2293:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB11997:
.LBB11994:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL2294:
.LBE11994:
.LBE11997:
	.loc 2 288 0
	movl	%eax, -448(%rbp)
	.loc 2 289 0
	jg	.L1163
.L1221:
	movl	-252(%rbp), %eax
	cmpl	$2, %eax
	jle	.L1296
.L1163:
	.loc 2 298 0
	leaq	-256(%rbp), %rsi
.LVL2295:
	leaq	-448(%rbp), %rdi
.LVL2296:
.LEHB36:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL2297:
.LEHE36:
.L1164:
	.loc 2 299 0
	movdqa	-240(%rbp), %xmm0
	.loc 2 303 0
	movq	-232(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -432(%rbp)
.LBE11989:
.LBE11988:
.LBB12004:
.LBB12005:
.LBB12006:
.LBB12007:
	.loc 2 366 0
	testq	%rax, %rax
.LBE12007:
.LBE12006:
.LBE12005:
.LBE12004:
.LBB12015:
.LBB12000:
	.loc 2 299 0
	movdqa	-224(%rbp), %xmm0
	movaps	%xmm0, -416(%rbp)
	movdqa	-208(%rbp), %xmm0
	movaps	%xmm0, -400(%rbp)
.LVL2298:
.LBE12000:
.LBE12015:
.LBB12016:
.LBB12014:
.LBB12012:
.LBB12010:
	.loc 2 366 0
	je	.L1214
	lock subl	$1, (%rax)
	jne	.L1214
	.loc 2 367 0
	leaq	-256(%rbp), %rdi
.LVL2299:
	call	_ZN2cv3Mat10deallocateEv
.LVL2300:
.L1214:
.LBB12008:
	.loc 2 369 0
	movl	-252(%rbp), %r8d
.LBE12008:
	.loc 2 368 0
	movq	$0, -208(%rbp)
	movq	$0, -216(%rbp)
	movq	$0, -224(%rbp)
	movq	$0, -240(%rbp)
.LVL2301:
.LBB12009:
	.loc 2 369 0
	testl	%r8d, %r8d
	jle	.L1170
	movq	-192(%rbp), %rdx
	xorl	%eax, %eax
.LVL2302:
	.p2align 4,,10
	.p2align 3
.L1171:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL2303:
	addq	$4, %rdx
	cmpl	%eax, -252(%rbp)
	jg	.L1171
.LVL2304:
.L1170:
.LBE12009:
.LBE12010:
.LBE12012:
	.loc 2 277 0
	movq	-184(%rbp), %rdi
	leaq	-256(%rbp), %rax
.LVL2305:
.LBB12013:
.LBB12011:
	.loc 2 371 0
	movq	$0, -232(%rbp)
.LVL2306:
.LBE12011:
.LBE12013:
	.loc 2 277 0
	addq	$80, %rax
.LVL2307:
	cmpq	%rax, %rdi
	je	.L1169
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL2308:
.L1169:
.LBE12014:
.LBE12016:
.LBB12017:
.LBB12018:
.LBB12019:
	.loc 2 709 0
	leaq	-160(%rbp), %rax
.LVL2309:
.LBE12019:
.LBE12018:
.LBB12022:
.LBB12023:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE12023:
.LBE12022:
.LBB12027:
.LBB12028:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm3
.LBE12028:
.LBE12027:
.LBB12031:
.LBB12032:
	.loc 2 353 0
	leaq	-464(%rbp), %rdx
.LBE12032:
.LBE12031:
.LBB12037:
.LBB12020:
	.loc 2 709 0
	addq	$8, %rax
.LVL2310:
.LBE12020:
.LBE12037:
.LBB12038:
.LBB12033:
	.loc 2 353 0
	leaq	-160(%rbp), %rdi
.LVL2311:
	movl	$21, %ecx
.LBE12033:
.LBE12038:
.LBB12039:
.LBB12021:
	.loc 2 709 0
	movq	%rax, -96(%rbp)
.LVL2312:
.LBE12021:
.LBE12039:
.LBB12040:
.LBB12024:
	.loc 2 738 0
	leaq	-160(%rbp), %rax
.LBE12024:
.LBE12040:
.LBB12041:
.LBB12034:
	.loc 2 353 0
	movl	$2, %esi
.LBE12034:
.LBE12041:
.LBB12042:
.LBB12025:
	.loc 2 738 0
	movaps	%xmm0, -80(%rbp)
.LVL2313:
	addq	$80, %rax
.LBE12025:
.LBE12042:
.LBB12043:
.LBB12029:
	.loc 2 62 0
	movq	$0, -112(%rbp)
	movq	$0, -120(%rbp)
	.loc 2 60 0
	movaps	%xmm3, -160(%rbp)
.LBE12029:
.LBE12043:
.LBB12044:
.LBB12026:
	.loc 2 738 0
	movq	%rax, -88(%rbp)
.LBE12026:
.LBE12044:
.LBB12045:
.LBB12035:
	.loc 2 352 0
	movl	-632(%rbp), %eax
.LBE12035:
.LBE12045:
.LBB12046:
.LBB12030:
	.loc 2 62 0
	movq	$0, -128(%rbp)
	movq	$0, -144(%rbp)
	.loc 2 63 0
	movq	$0, -136(%rbp)
	.loc 2 64 0
	movq	$0, -104(%rbp)
.LVL2314:
.LBE12030:
.LBE12046:
.LBB12047:
.LBB12036:
	.loc 2 352 0
	movl	%eax, -464(%rbp)
	movl	-616(%rbp), %eax
	movl	%eax, -460(%rbp)
.LEHB37:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL2315:
.LEHE37:
.LBE12036:
.LBE12047:
.LBE12017:
.LBB12048:
.LBB12049:
	.loc 2 285 0
	movq	-136(%rbp), %rax
	testq	%rax, %rax
	je	.L1172
	.loc 2 286 0
	lock addl	$1, (%rax)
.L1172:
.LVL2316:
.LBB12050:
.LBB12051:
	.loc 2 366 0
	movq	-328(%rbp), %rax
	testq	%rax, %rax
	je	.L1215
	lock subl	$1, (%rax)
	je	.L1297
.L1215:
.LBB12052:
	.loc 2 369 0
	movl	-348(%rbp), %eax
.LBE12052:
	.loc 2 368 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
.LVL2317:
.LBB12053:
	.loc 2 369 0
	testl	%eax, %eax
	jle	.L1298
	movq	-288(%rbp), %rdx
	xorl	%eax, %eax
.LVL2318:
	.p2align 4,,10
	.p2align 3
.L1175:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-348(%rbp), %ecx
	addl	$1, %eax
.LVL2319:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L1175
.LBE12053:
.LBE12051:
.LBE12050:
	.loc 2 288 0
	movl	-160(%rbp), %eax
.LVL2320:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB12057:
.LBB12054:
	.loc 2 371 0
	movq	$0, -328(%rbp)
.LVL2321:
.LBE12054:
.LBE12057:
	.loc 2 288 0
	movl	%eax, -352(%rbp)
	.loc 2 289 0
	jg	.L1176
.L1222:
	movl	-156(%rbp), %eax
	cmpl	$2, %eax
	jle	.L1299
.L1176:
	.loc 2 298 0
	leaq	-160(%rbp), %rsi
.LVL2322:
	leaq	-352(%rbp), %rdi
.LVL2323:
.LEHB38:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL2324:
.LEHE38:
.L1177:
	.loc 2 299 0
	movdqa	-144(%rbp), %xmm0
	.loc 2 303 0
	movq	-136(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -336(%rbp)
.LBE12049:
.LBE12048:
.LBB12064:
.LBB12065:
.LBB12066:
.LBB12067:
	.loc 2 366 0
	testq	%rax, %rax
.LBE12067:
.LBE12066:
.LBE12065:
.LBE12064:
.LBB12075:
.LBB12060:
	.loc 2 299 0
	movdqa	-128(%rbp), %xmm0
	movaps	%xmm0, -320(%rbp)
	movdqa	-112(%rbp), %xmm0
	movaps	%xmm0, -304(%rbp)
.LVL2325:
.LBE12060:
.LBE12075:
.LBB12076:
.LBB12074:
.LBB12072:
.LBB12070:
	.loc 2 366 0
	je	.L1218
	lock subl	$1, (%rax)
	jne	.L1218
	.loc 2 367 0
	leaq	-160(%rbp), %rdi
.LVL2326:
	call	_ZN2cv3Mat10deallocateEv
.LVL2327:
.L1218:
.LBB12068:
	.loc 2 369 0
	movl	-156(%rbp), %edi
.LBE12068:
	.loc 2 368 0
	movq	$0, -112(%rbp)
	movq	$0, -120(%rbp)
	movq	$0, -128(%rbp)
	movq	$0, -144(%rbp)
.LVL2328:
.LBB12069:
	.loc 2 369 0
	testl	%edi, %edi
	jle	.L1183
	movq	-96(%rbp), %rdx
	xorl	%eax, %eax
.LVL2329:
	.p2align 4,,10
	.p2align 3
.L1184:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL2330:
	addq	$4, %rdx
	cmpl	%eax, -156(%rbp)
	jg	.L1184
.LVL2331:
.L1183:
.LBE12069:
.LBE12070:
.LBE12072:
	.loc 2 277 0
	movq	-88(%rbp), %rdi
	leaq	-160(%rbp), %rax
.LVL2332:
.LBB12073:
.LBB12071:
	.loc 2 371 0
	movq	$0, -136(%rbp)
.LVL2333:
.LBE12071:
.LBE12073:
	.loc 2 277 0
	addq	$80, %rax
.LVL2334:
	cmpq	%rax, %rdi
	je	.L1182
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL2335:
.L1182:
.LBE12074:
.LBE12076:
.LBB12077:
	.loc 1 99 0
	testl	%r12d, %r12d
	jle	.L1190
	movslq	%ebx, %rax
	movl	%ebx, -704(%rbp)
	xorl	%r15d, %r15d
	movq	%rax, -648(%rbp)
.LVL2336:
	leaq	(%rax,%rax,2), %r14
	movslq	-624(%rbp), %rax
	movq	%r13, %rbx
.LVL2337:
	leaq	(%rax,%rax,2), %rax
	movq	%rax, %r13
.LVL2338:
	.p2align 4,,10
	.p2align 3
.L1188:
.LBB12078:
.LBB12079:
.LBB12080:
	.loc 2 430 0 discriminator 2
	movq	-376(%rbp), %rax
.LBE12080:
.LBE12079:
.LBB12081:
.LBB12082:
	.loc 5 53 0 discriminator 2
	movq	%r15, %rdi
	movq	%r15, %rsi
	movq	%r13, %rdx
	addq	$1, %r15
.LVL2339:
	imulq	(%rax), %rdi
.LBE12082:
.LBE12081:
.LBB12084:
.LBB12085:
	.loc 2 436 0 discriminator 2
	movq	72(%rbx), %rax
.LBE12085:
.LBE12084:
.LBB12086:
.LBB12083:
	.loc 5 53 0 discriminator 2
	imulq	(%rax), %rsi
.LVL2340:
	addq	%r14, %rdi
	addq	-432(%rbp), %rdi
	addq	16(%rbx), %rsi
	call	memcpy
.LVL2341:
.LBE12083:
.LBE12086:
.LBE12078:
	.loc 1 99 0 discriminator 2
	cmpl	%r15d, %r12d
	jg	.L1188
	movl	-640(%rbp), %esi
	movl	-704(%rbp), %ebx
.LVL2342:
	.loc 1 99 0 is_stmt 0
	xorl	%r10d, %r10d
	movl	%esi, %eax
	subl	%ebx, %eax
	cltq
	leaq	-3(%rax,%rax,2), %r13
	leal	-1(%rbx), %eax
	leaq	3(%rax,%rax,2), %r11
	movslq	%esi, %rax
	movq	-648(%rbp), %rsi
	leaq	(%rax,%rax,2), %r9
	movq	%rsi, %rax
	negq	%rax
	leaq	(%rsi,%rax,4), %rdi
	.p2align 4,,10
	.p2align 3
.L1191:
.LVL2343:
.LBE12077:
.LBB12087:
.LBB12088:
.LBB12089:
.LBB12090:
	.loc 2 430 0 is_stmt 1
	movq	-376(%rbp), %rax
	movq	%r10, %rcx
	imulq	(%rax), %rcx
	movq	%rcx, %rax
	addq	-432(%rbp), %rax
.LVL2344:
.LBE12090:
.LBE12089:
.LBB12091:
	.loc 1 108 0
	testl	%ebx, %ebx
	leaq	(%rax,%r14), %rsi
	leaq	(%rax,%r13), %rcx
	leaq	(%rax,%r11), %r15
	jle	.L1192
.LVL2345:
	.p2align 4,,10
	.p2align 3
.L1193:
	.loc 1 109 0 discriminator 2
	movzwl	(%rsi), %edx
	movw	%dx, (%rax)
	movzbl	2(%rsi), %edx
	movb	%dl, 2(%rax)
	.loc 1 110 0 discriminator 2
	movzwl	(%rcx), %r8d
	leaq	(%rax,%r9), %rdx
	addq	$3, %rax
	.loc 1 108 0 discriminator 2
	cmpq	%r15, %rax
	.loc 1 110 0 discriminator 2
	movw	%r8w, (%rdx,%rdi)
	movzbl	2(%rcx), %r8d
	movb	%r8b, 2(%rdx,%rdi)
	.loc 1 108 0 discriminator 2
	jne	.L1193
.L1192:
.LVL2346:
	addq	$1, %r10
.LVL2347:
.LBE12091:
.LBE12088:
	.loc 1 106 0
	cmpl	%r10d, %r12d
	jg	.L1191
.LVL2348:
.L1190:
.LBE12087:
	.loc 1 116 0
	testb	$15, -616(%rbp)
	jne	.L1300
.LBB12092:
	.loc 1 120 0
	movq	-664(%rbp), %rax
	pxor	%xmm0, %xmm0
	leaq	-592(%rbp), %rsi
	xorl	%ecx, %ecx
	movl	$4, %edx
	movl	$_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.0, %edi
	movl	%ebx, -520(%rbp)
	movl	%r12d, -512(%rbp)
	movq	%rax, -552(%rbp)
	movq	-656(%rbp), %rax
	movaps	%xmm0, -592(%rbp)
	movl	$3, -496(%rbp)
	movq	%rax, -560(%rbp)
	leaq	-600(%rbp), %rax
	movq	%rax, -576(%rbp)
	leaq	-596(%rbp), %rax
	movq	%rax, -568(%rbp)
	leaq	-448(%rbp), %rax
	movq	%rax, -544(%rbp)
	leaq	-352(%rbp), %rax
	movq	%rax, -536(%rbp)
	movq	-672(%rbp), %rax
	movq	%rax, -528(%rbp)
	movl	-676(%rbp), %eax
	movl	%eax, -516(%rbp)
	movl	-616(%rbp), %eax
	movl	%eax, -508(%rbp)
	movl	-624(%rbp), %eax
	movl	%eax, -504(%rbp)
	movl	-632(%rbp), %eax
	movl	%eax, -500(%rbp)
	call	GOMP_parallel
.LVL2349:
.LBE12092:
.LBB12093:
.LBB12094:
.LBB12095:
.LBB12096:
	.loc 2 366 0
	movq	-328(%rbp), %rax
	testq	%rax, %rax
	je	.L1219
	lock subl	$1, (%rax)
	jne	.L1219
	.loc 2 367 0
	leaq	-352(%rbp), %rdi
.LVL2350:
	call	_ZN2cv3Mat10deallocateEv
.LVL2351:
.L1219:
.LBB12097:
	.loc 2 369 0
	movl	-348(%rbp), %esi
.LBE12097:
	.loc 2 368 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
.LVL2352:
.LBB12098:
	.loc 2 369 0
	testl	%esi, %esi
	jle	.L1199
	movq	-288(%rbp), %rdx
	xorl	%eax, %eax
.LVL2353:
	.p2align 4,,10
	.p2align 3
.L1200:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL2354:
	addq	$4, %rdx
	cmpl	%eax, -348(%rbp)
	jg	.L1200
.LVL2355:
.L1199:
.LBE12098:
.LBE12096:
.LBE12095:
	.loc 2 277 0
	movq	-280(%rbp), %rdi
	leaq	-352(%rbp), %rax
.LVL2356:
.LBB12100:
.LBB12099:
	.loc 2 371 0
	movq	$0, -328(%rbp)
.LVL2357:
.LBE12099:
.LBE12100:
	.loc 2 277 0
	addq	$80, %rax
.LVL2358:
	cmpq	%rax, %rdi
	je	.L1198
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL2359:
.L1198:
.LBE12094:
.LBE12093:
.LBB12101:
.LBB12102:
.LBB12103:
.LBB12104:
	.loc 2 366 0
	movq	-424(%rbp), %rax
	testq	%rax, %rax
	je	.L1220
	lock subl	$1, (%rax)
	jne	.L1220
	.loc 2 367 0
	leaq	-448(%rbp), %rdi
.LVL2360:
	call	_ZN2cv3Mat10deallocateEv
.LVL2361:
.L1220:
.LBB12105:
	.loc 2 369 0
	movl	-444(%rbp), %ecx
.LBE12105:
	.loc 2 368 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
.LVL2362:
.LBB12106:
	.loc 2 369 0
	testl	%ecx, %ecx
	jle	.L1206
	movq	-384(%rbp), %rdx
	xorl	%eax, %eax
.LVL2363:
	.p2align 4,,10
	.p2align 3
.L1207:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL2364:
	addq	$4, %rdx
	cmpl	%eax, -444(%rbp)
	jg	.L1207
.LVL2365:
.L1206:
.LBE12106:
.LBE12104:
.LBE12103:
	.loc 2 277 0
	movq	-376(%rbp), %rdi
	leaq	-448(%rbp), %rax
.LVL2366:
.LBB12108:
.LBB12107:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL2367:
.LBE12107:
.LBE12108:
	.loc 2 277 0
	addq	$80, %rax
.LVL2368:
	cmpq	%rax, %rdi
	je	.L1118
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL2369:
.L1118:
.LBE12102:
.LBE12101:
	.loc 1 293 0
	movq	-56(%rbp), %rax
	xorq	%fs:40, %rax
	jne	.L1301
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
.LVL2370:
	ret
.LVL2371:
.L1299:
	.cfi_restore_state
.LBB12109:
.LBB12061:
	.loc 2 291 0
	movl	%eax, -348(%rbp)
	.loc 2 292 0
	movl	-152(%rbp), %eax
	movq	-88(%rbp), %rdx
	movl	%eax, -344(%rbp)
	.loc 2 293 0
	movl	-148(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -340(%rbp)
	movq	-280(%rbp), %rax
.LVL2372:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL2373:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L1177
.LVL2374:
.L1296:
.LBE12061:
.LBE12109:
.LBB12110:
.LBB12001:
	.loc 2 291 0
	movl	%eax, -444(%rbp)
	.loc 2 292 0
	movl	-248(%rbp), %eax
	movq	-184(%rbp), %rdx
	movl	%eax, -440(%rbp)
	.loc 2 293 0
	movl	-244(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -436(%rbp)
	movq	-376(%rbp), %rax
.LVL2375:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL2376:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L1164
.LVL2377:
.L1292:
	testl	%edx, %edx
	jne	.L1149
.LBE12001:
.LBE12110:
.LBB12111:
	.loc 1 72 0
	xorl	%ecx, %ecx
	jmp	.L1150
.LVL2378:
.L1289:
	testl	%edx, %edx
	jne	.L1134
.LBE12111:
.LBB12112:
	.loc 1 65 0
	xorl	%ecx, %ecx
	jmp	.L1135
.LVL2379:
.L1297:
.LBE12112:
.LBB12113:
.LBB12062:
.LBB12058:
.LBB12055:
	.loc 2 367 0
	leaq	-352(%rbp), %rdi
.LVL2380:
.LEHB39:
	call	_ZN2cv3Mat10deallocateEv
.LVL2381:
.LEHE39:
	jmp	.L1215
.LVL2382:
.L1294:
.LBE12055:
.LBE12058:
.LBE12062:
.LBE12113:
.LBB12114:
.LBB12002:
.LBB11998:
.LBB11995:
	leaq	-448(%rbp), %rdi
.LVL2383:
.LEHB40:
	call	_ZN2cv3Mat10deallocateEv
.LVL2384:
.LEHE40:
	jmp	.L1213
.LVL2385:
.L1298:
.LBE11995:
.LBE11998:
.LBE12002:
.LBE12114:
.LBB12115:
.LBB12063:
	.loc 2 288 0
	movl	-160(%rbp), %eax
.LBB12059:
.LBB12056:
	.loc 2 371 0
	movq	$0, -328(%rbp)
.LVL2386:
.LBE12056:
.LBE12059:
	.loc 2 288 0
	movl	%eax, -352(%rbp)
	jmp	.L1222
.LVL2387:
.L1295:
.LBE12063:
.LBE12115:
.LBB12116:
.LBB12003:
	movl	-256(%rbp), %eax
.LBB11999:
.LBB11996:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL2388:
.LBE11996:
.LBE11999:
	.loc 2 288 0
	movl	%eax, -448(%rbp)
	jmp	.L1221
.LVL2389:
.L1119:
.LBE12003:
.LBE12116:
	.loc 1 47 0 discriminator 3
	movl	$_ZZ12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, %ecx
	movl	$47, %edx
	movl	$.LC45, %esi
	movl	$.LC68, %edi
.LVL2390:
	call	__assert_fail
.LVL2391:
.L1235:
	movq	%rax, %rbx
.LVL2392:
	jmp	.L1208
.LVL2393:
.L1234:
	movq	%rax, %rbx
.LVL2394:
	jmp	.L1209
.LVL2395:
.L1293:
	.loc 1 83 0 discriminator 1
	movl	$_ZZ12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, %ecx
	movl	$83, %edx
.LVL2396:
	movl	$.LC45, %esi
	movl	$.LC70, %edi
	call	__assert_fail
.LVL2397:
.L1208:
	.loc 1 95 0 discriminator 2
	leaq	-256(%rbp), %rdi
.LVL2398:
	call	_ZN2cv3MatD1Ev
.LVL2399:
.L1209:
	.loc 1 85 0
	leaq	-352(%rbp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL2400:
	.loc 1 84 0
	leaq	-448(%rbp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL2401:
	movq	%rbx, %rdi
.LEHB41:
	call	_Unwind_Resume
.LVL2402:
.LEHE41:
.L1121:
	.loc 1 54 0 discriminator 3
	movl	$_ZZ12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, %ecx
.LVL2403:
	movl	$54, %edx
.LVL2404:
	movl	$.LC45, %esi
	movl	$.LC69, %edi
.LVL2405:
	call	__assert_fail
.LVL2406:
.L1300:
	.loc 1 116 0 discriminator 1
	movl	$_ZZ12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, %ecx
	movl	$116, %edx
	movl	$.LC45, %esi
	movl	$.LC71, %edi
	call	__assert_fail
.LVL2407:
.L1236:
	movq	%rax, %rbx
.LVL2408:
	jmp	.L1210
.LVL2409:
.L1301:
	.loc 1 293 0
	call	__stack_chk_fail
.LVL2410:
.L1210:
	.loc 1 96 0 discriminator 2
	leaq	-160(%rbp), %rdi
.LVL2411:
	call	_ZN2cv3MatD1Ev
.LVL2412:
	jmp	.L1209
	.cfi_endproc
.LFE11596:
	.section	.gcc_except_table
.LLSDA11596:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE11596-.LLSDACSB11596
.LLSDACSB11596:
	.uleb128 .LEHB35-.LFB11596
	.uleb128 .LEHE35-.LEHB35
	.uleb128 .L1234-.LFB11596
	.uleb128 0
	.uleb128 .LEHB36-.LFB11596
	.uleb128 .LEHE36-.LEHB36
	.uleb128 .L1235-.LFB11596
	.uleb128 0
	.uleb128 .LEHB37-.LFB11596
	.uleb128 .LEHE37-.LEHB37
	.uleb128 .L1234-.LFB11596
	.uleb128 0
	.uleb128 .LEHB38-.LFB11596
	.uleb128 .LEHE38-.LEHB38
	.uleb128 .L1236-.LFB11596
	.uleb128 0
	.uleb128 .LEHB39-.LFB11596
	.uleb128 .LEHE39-.LEHB39
	.uleb128 .L1236-.LFB11596
	.uleb128 0
	.uleb128 .LEHB40-.LFB11596
	.uleb128 .LEHE40-.LEHB40
	.uleb128 .L1235-.LFB11596
	.uleb128 0
	.uleb128 .LEHB41-.LFB11596
	.uleb128 .LEHE41-.LEHB41
	.uleb128 0
	.uleb128 0
.LLSDACSE11596:
	.section	.text._Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
	.size	_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd, .-_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd
	.section	.text.unlikely._Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LCOLDE72:
	.section	.text._Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LHOTE72:
	.section	.text.unlikely._Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LCOLDB73:
	.section	.text._Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LHOTB73:
	.p2align 4,,15
	.weak	_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd
	.type	_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd, @function
_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd:
.LFB11597:
	.loc 1 40 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA11597
.LVL2413:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movapd	%xmm0, %xmm3
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$680, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	.loc 1 40 0
	movq	%rsi, -672(%rbp)
	movsd	.LC67(%rip), %xmm2
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
.LVL2414:
	mulsd	%xmm2, %xmm3
	mulsd	%xmm1, %xmm2
	cvttsd2si	%xmm3, %eax
	leal	1(%rax,%rax), %ecx
	movl	(%rdx), %eax
	cmpl	%eax, %ecx
	cmovg	%eax, %ecx
.LVL2415:
	cvttsd2si	%xmm2, %eax
	.loc 1 45 0
	movl	%ecx, -600(%rbp)
.LVL2416:
	leal	1(%rax,%rax), %esi
.LVL2417:
	movl	4(%rdx), %eax
	cmpl	%eax, %esi
	cmovle	%esi, %eax
.LVL2418:
	.loc 1 47 0
	movl	%ecx, %esi
	shrl	$31, %esi
	.loc 1 46 0
	movl	%eax, -596(%rbp)
	.loc 1 47 0
	leal	(%rcx,%rsi), %edx
.LVL2419:
	andl	$1, %edx
	subl	%esi, %edx
	cmpl	$1, %edx
	jne	.L1303
	.loc 1 47 0 is_stmt 0 discriminator 2
	movl	%eax, %esi
	shrl	$31, %esi
	leal	(%rax,%rsi), %edx
	andl	$1, %edx
	subl	%esi, %edx
	cmpl	$1, %edx
	jne	.L1303
	.loc 1 49 0 is_stmt 1
	movl	%eax, %edx
	.loc 1 48 0
	movl	%ecx, %ebx
	.loc 1 49 0
	shrl	$31, %edx
	.loc 1 48 0
	shrl	$31, %ebx
	.loc 1 49 0
	addl	%edx, %eax
	.loc 1 48 0
	addl	%ecx, %ebx
	.loc 1 49 0
	sarl	%eax
	.loc 1 48 0
	sarl	%ebx
.LVL2420:
	.loc 1 51 0
	leal	1(%rax), %ecx
	.loc 1 49 0
	movl	%eax, -676(%rbp)
.LVL2421:
	.loc 1 50 0
	leal	1(%rbx), %edx
	.loc 1 54 0
	pxor	%xmm7, %xmm7
	.loc 1 52 0
	movslq	%ecx, %rax
.LVL2422:
	.loc 1 50 0
	movl	%edx, -600(%rbp)
	.loc 1 51 0
	movl	%ecx, -596(%rbp)
.LVL2423:
	.loc 1 52 0
	leaq	18(,%rax,4), %rax
	andq	$-16, %rax
	subq	%rax, %rsp
	leaq	3(%rsp), %r14
	shrq	$2, %r14
	leaq	0(,%r14,4), %rax
	movq	%rax, -656(%rbp)
.LVL2424:
	.loc 1 53 0
	movslq	%edx, %rax
.LVL2425:
	leaq	18(,%rax,4), %rax
	andq	$-16, %rax
	subq	%rax, %rsp
	leaq	3(%rsp), %r12
	shrq	$2, %r12
	.loc 1 54 0
	ucomisd	%xmm7, %xmm0
	.loc 1 53 0
	leaq	0(,%r12,4), %rax
	movq	%rax, -664(%rbp)
.LVL2426:
	.loc 1 54 0
	jbe	.L1305
	.loc 1 54 0 is_stmt 0 discriminator 2
	ucomisd	%xmm7, %xmm1
	jbe	.L1305
	.loc 1 55 0 is_stmt 1
	movsd	.LC38(%rip), %xmm3
	movapd	%xmm0, %xmm4
	movsd	.LC15(%rip), %xmm2
.LBB12211:
	.loc 1 60 0
	testl	%ecx, %ecx
.LBE12211:
	.loc 1 55 0
	mulsd	%xmm3, %xmm4
	movq	%rdi, %r13
	.loc 1 56 0
	mulsd	%xmm1, %xmm3
	movapd	%xmm2, %xmm6
	.loc 1 55 0
	movapd	%xmm2, %xmm5
	divsd	%xmm4, %xmm5
	.loc 1 56 0
	divsd	%xmm3, %xmm6
	.loc 1 57 0
	movapd	%xmm0, %xmm3
	addsd	%xmm0, %xmm3
	.loc 1 55 0
	movsd	%xmm5, -624(%rbp)
.LVL2427:
	.loc 1 57 0
	mulsd	%xmm3, %xmm0
.LVL2428:
	movapd	%xmm2, %xmm3
	.loc 1 56 0
	movsd	%xmm6, -632(%rbp)
.LVL2429:
	.loc 1 57 0
	divsd	%xmm0, %xmm3
	.loc 1 58 0
	movapd	%xmm1, %xmm0
	addsd	%xmm1, %xmm0
	mulsd	%xmm0, %xmm1
.LVL2430:
	.loc 1 57 0
	movsd	%xmm3, -640(%rbp)
.LVL2431:
	.loc 1 58 0
	divsd	%xmm1, %xmm2
	movsd	%xmm2, -648(%rbp)
.LVL2432:
.LBB12212:
	.loc 1 60 0
	jle	.L1308
	xorl	%r15d, %r15d
	movl	%ebx, -680(%rbp)
	pxor	%xmm1, %xmm1
	movl	%r15d, %ebx
.LVL2433:
	movq	-656(%rbp), %r15
	jmp	.L1314
.LVL2434:
	.p2align 4,,10
	.p2align 3
.L1472:
	.loc 1 62 0 discriminator 1
	addsd	%xmm0, %xmm0
	.loc 1 60 0 discriminator 1
	movl	-596(%rbp), %eax
	addl	$1, %ebx
.LVL2435:
	cmpl	%ebx, %eax
	.loc 1 62 0 discriminator 1
	addsd	%xmm0, %xmm1
.LVL2436:
	.loc 1 60 0 discriminator 1
	jle	.L1471
.LVL2437:
.L1314:
	.loc 1 61 0
	movl	%ebx, %eax
	pxor	%xmm0, %xmm0
	negl	%eax
	movsd	%xmm1, -616(%rbp)
.LVL2438:
	imull	%ebx, %eax
	cvtsi2sd	%eax, %xmm0
	mulsd	-648(%rbp), %xmm0
	call	exp
.LVL2439:
	mulsd	-632(%rbp), %xmm0
	movslq	%ebx, %rax
	.loc 1 62 0
	testl	%ebx, %ebx
	movsd	-616(%rbp), %xmm1
	.loc 1 61 0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, (%r15,%rax,4)
	.loc 1 62 0
	cvtss2sd	%xmm0, %xmm0
	jg	.L1472
.LVL2440:
	.loc 1 60 0
	movl	-596(%rbp), %eax
	addl	$1, %ebx
.LVL2441:
	.loc 1 63 0
	addsd	%xmm0, %xmm1
.LVL2442:
	.loc 1 60 0
	cmpl	%ebx, %eax
	jg	.L1314
.L1471:
.LBE12212:
.LBB12213:
	.loc 1 65 0
	testl	%eax, %eax
	movl	-680(%rbp), %ebx
.LVL2443:
	jle	.L1321
	movq	-656(%rbp), %rdx
	andl	$15, %edx
	shrq	$2, %rdx
	negq	%rdx
	andl	$3, %edx
	cmpl	%eax, %edx
	cmova	%eax, %edx
	cmpl	$4, %eax
	jg	.L1473
	movl	%eax, %edx
.L1318:
	pxor	%xmm0, %xmm0
	cmpl	$1, %edx
	movl	$1, %ecx
	pxor	%xmm4, %xmm4
	cvtss2sd	0(,%r14,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm4
	movss	%xmm4, 0(,%r14,4)
.LVL2444:
	je	.L1320
	pxor	%xmm0, %xmm0
	cmpl	$2, %edx
	movl	$2, %ecx
	pxor	%xmm5, %xmm5
	cvtss2sd	4(,%r14,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm5
	movss	%xmm5, 4(,%r14,4)
.LVL2445:
	je	.L1320
	pxor	%xmm0, %xmm0
	cmpl	$3, %edx
	movl	$3, %ecx
	pxor	%xmm6, %xmm6
	cvtss2sd	8(,%r14,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm6
	movss	%xmm6, 8(,%r14,4)
.LVL2446:
	je	.L1320
	pxor	%xmm0, %xmm0
	movl	$4, %ecx
	pxor	%xmm7, %xmm7
	cvtss2sd	12(,%r14,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm7
	movss	%xmm7, 12(,%r14,4)
.LVL2447:
.L1320:
	cmpl	%edx, %eax
	je	.L1321
.L1319:
	movl	%eax, %r8d
	leal	-1(%rax), %edi
	movl	%edx, %r10d
	subl	%edx, %r8d
	leal	-4(%r8), %esi
	subl	%edx, %edi
	shrl	$2, %esi
	addl	$1, %esi
	cmpl	$2, %edi
	leal	0(,%rsi,4), %r9d
	jbe	.L1322
	movq	-656(%rbp), %rdi
	movddup	%xmm1, %xmm3
	xorl	%edx, %edx
	leaq	(%rdi,%r10,4), %r10
	xorl	%edi, %edi
.L1324:
	.loc 1 65 0 is_stmt 0 discriminator 2
	movaps	(%r10,%rdx), %xmm2
	addl	$1, %edi
	movhps	%xmm2, -704(%rbp)
	cvtps2pd	%xmm2, %xmm0
	cvtps2pd	-704(%rbp), %xmm2
	divpd	%xmm3, %xmm0
	divpd	%xmm3, %xmm2
	cvtpd2ps	%xmm0, %xmm0
	cvtpd2ps	%xmm2, %xmm2
	movlhps	%xmm2, %xmm0
	movaps	%xmm0, (%r10,%rdx)
	addq	$16, %rdx
	cmpl	%edi, %esi
	ja	.L1324
	addl	%r9d, %ecx
	cmpl	%r8d, %r9d
	je	.L1321
.L1322:
.LVL2448:
	movq	-656(%rbp), %rsi
	movslq	%ecx, %rdx
	.loc 1 65 0
	pxor	%xmm0, %xmm0
	pxor	%xmm4, %xmm4
	leaq	(%rsi,%rdx,4), %rdx
	cvtss2sd	(%rdx), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm4
	movss	%xmm4, (%rdx)
	leal	1(%rcx), %edx
.LVL2449:
	cmpl	%edx, %eax
	jle	.L1321
	movslq	%edx, %rdx
	pxor	%xmm0, %xmm0
	leaq	(%rsi,%rdx,4), %rdx
.LVL2450:
	pxor	%xmm5, %xmm5
	addl	$2, %ecx
.LVL2451:
	cvtss2sd	(%rdx), %xmm0
	cmpl	%ecx, %eax
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm5
	movss	%xmm5, (%rdx)
.LVL2452:
	jle	.L1321
	movslq	%ecx, %rcx
	pxor	%xmm0, %xmm0
	leaq	(%rsi,%rcx,4), %rax
	pxor	%xmm7, %xmm7
	cvtss2sd	(%rax), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm7
	movss	%xmm7, (%rax)
.L1321:
	movl	-600(%rbp), %edx
.LVL2453:
.L1308:
.LBE12213:
.LBB12214:
	.loc 1 67 0 is_stmt 1 discriminator 1
	xorl	%r14d, %r14d
	testl	%edx, %edx
	movq	-664(%rbp), %r15
	pxor	%xmm1, %xmm1
	jg	.L1441
	jmp	.L1327
.LVL2454:
	.p2align 4,,10
	.p2align 3
.L1475:
	.loc 1 69 0 discriminator 1
	addsd	%xmm0, %xmm0
	.loc 1 67 0 discriminator 1
	movl	-600(%rbp), %eax
	addl	$1, %r14d
.LVL2455:
	cmpl	%r14d, %eax
	.loc 1 69 0 discriminator 1
	addsd	%xmm0, %xmm1
.LVL2456:
	.loc 1 67 0 discriminator 1
	jle	.L1474
.LVL2457:
.L1441:
	.loc 1 68 0
	movl	%r14d, %eax
	pxor	%xmm0, %xmm0
	negl	%eax
	movsd	%xmm1, -616(%rbp)
.LVL2458:
	imull	%r14d, %eax
	cvtsi2sd	%eax, %xmm0
	mulsd	-640(%rbp), %xmm0
	call	exp
.LVL2459:
	mulsd	-624(%rbp), %xmm0
	movslq	%r14d, %rax
	.loc 1 69 0
	testl	%r14d, %r14d
	movsd	-616(%rbp), %xmm1
	.loc 1 68 0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, (%r15,%rax,4)
	.loc 1 69 0
	cvtss2sd	%xmm0, %xmm0
	jg	.L1475
.LVL2460:
	.loc 1 67 0
	movl	-600(%rbp), %eax
	addl	$1, %r14d
.LVL2461:
	.loc 1 70 0
	addsd	%xmm0, %xmm1
.LVL2462:
	.loc 1 67 0
	cmpl	%r14d, %eax
	jg	.L1441
.L1474:
.LVL2463:
.LBE12214:
.LBB12215:
	.loc 1 72 0 discriminator 3
	testl	%eax, %eax
	jle	.L1327
	movq	-664(%rbp), %rdx
	andl	$15, %edx
	shrq	$2, %rdx
	negq	%rdx
	andl	$3, %edx
	cmpl	%eax, %edx
	cmova	%eax, %edx
	cmpl	$4, %eax
	jg	.L1476
	.loc 1 72 0 is_stmt 0
	movl	%eax, %edx
.L1333:
	pxor	%xmm0, %xmm0
	cmpl	$1, %edx
	movl	$1, %ecx
	pxor	%xmm4, %xmm4
	cvtss2sd	0(,%r12,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm4
	movss	%xmm4, 0(,%r12,4)
.LVL2464:
	je	.L1335
	pxor	%xmm0, %xmm0
	cmpl	$2, %edx
	movl	$2, %ecx
	pxor	%xmm5, %xmm5
	cvtss2sd	4(,%r12,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm5
	movss	%xmm5, 4(,%r12,4)
.LVL2465:
	je	.L1335
	pxor	%xmm0, %xmm0
	cmpl	$3, %edx
	movl	$3, %ecx
	pxor	%xmm6, %xmm6
	cvtss2sd	8(,%r12,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm6
	movss	%xmm6, 8(,%r12,4)
.LVL2466:
	je	.L1335
	pxor	%xmm0, %xmm0
	movl	$4, %ecx
	pxor	%xmm3, %xmm3
	cvtss2sd	12(,%r12,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm3
	movss	%xmm3, 12(,%r12,4)
.LVL2467:
.L1335:
	cmpl	%edx, %eax
	je	.L1327
.L1334:
	movl	%eax, %r8d
	leal	-1(%rax), %edi
	movl	%edx, %r10d
	subl	%edx, %r8d
	leal	-4(%r8), %esi
	subl	%edx, %edi
	shrl	$2, %esi
	addl	$1, %esi
	cmpl	$2, %edi
	leal	0(,%rsi,4), %r9d
	jbe	.L1337
	movq	-664(%rbp), %rdi
	movddup	%xmm1, %xmm3
	xorl	%edx, %edx
	leaq	(%rdi,%r10,4), %r10
	xorl	%edi, %edi
.L1339:
	.loc 1 72 0 discriminator 2
	movaps	(%r10,%rdx), %xmm2
	addl	$1, %edi
	movhps	%xmm2, -720(%rbp)
	cvtps2pd	%xmm2, %xmm0
	cvtps2pd	-720(%rbp), %xmm2
	divpd	%xmm3, %xmm0
	divpd	%xmm3, %xmm2
	cvtpd2ps	%xmm0, %xmm0
	cvtpd2ps	%xmm2, %xmm2
	movlhps	%xmm2, %xmm0
	movaps	%xmm0, (%r10,%rdx)
	addq	$16, %rdx
	cmpl	%esi, %edi
	jb	.L1339
	addl	%r9d, %ecx
	cmpl	%r8d, %r9d
	je	.L1327
.L1337:
.LVL2468:
	movq	-664(%rbp), %rsi
	movslq	%ecx, %rdx
	.loc 1 72 0
	pxor	%xmm0, %xmm0
	pxor	%xmm3, %xmm3
	leaq	(%rsi,%rdx,4), %rdx
	cvtss2sd	(%rdx), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm3
	movss	%xmm3, (%rdx)
	leal	1(%rcx), %edx
.LVL2469:
	cmpl	%edx, %eax
	jle	.L1327
	movslq	%edx, %rdx
	pxor	%xmm0, %xmm0
	leaq	(%rsi,%rdx,4), %rdx
.LVL2470:
	pxor	%xmm7, %xmm7
	addl	$2, %ecx
.LVL2471:
	cvtss2sd	(%rdx), %xmm0
	cmpl	%ecx, %eax
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm7
	movss	%xmm7, (%rdx)
.LVL2472:
	jle	.L1327
	movslq	%ecx, %rcx
	pxor	%xmm0, %xmm0
	leaq	(%rsi,%rcx,4), %rax
	pxor	%xmm6, %xmm6
	cvtss2sd	(%rax), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm6
	movss	%xmm6, (%rax)
.LVL2473:
.L1327:
.LBE12215:
	.loc 1 75 0 is_stmt 1
	movl	12(%r13), %eax
	.loc 1 74 0
	movl	8(%r13), %r12d
.LVL2474:
	.loc 1 78 0
	testb	$15, %al
	.loc 1 75 0
	movl	%eax, -624(%rbp)
.LVL2475:
	.loc 1 78 0
	je	.L1463
	movl	%eax, %ecx
	.loc 1 78 0 is_stmt 0 discriminator 1
	addl	$15, %eax
.LVL2476:
	testl	%ecx, %ecx
	cmovns	%ecx, %eax
	andl	$-16, %eax
	addl	$16, %eax
.LVL2477:
.L1463:
	movl	%eax, %ecx
.LVL2478:
	movl	%eax, -616(%rbp)
.LVL2479:
	.loc 1 80 0 is_stmt 1 discriminator 1
	leal	(%rbx,%rbx), %eax
	leal	(%rcx,%rax), %edx
.LVL2480:
	.loc 1 81 0 discriminator 1
	addl	-624(%rbp), %eax
	movl	%eax, -640(%rbp)
.LVL2481:
	.loc 1 82 0 discriminator 1
	movl	-676(%rbp), %eax
.LVL2482:
	leal	(%r12,%rax,2), %eax
	movl	%eax, -632(%rbp)
.LVL2483:
	.loc 1 83 0 discriminator 1
	movq	-672(%rbp), %rax
.LVL2484:
	movq	8(%rax), %rax
	cmpq	%rax, 8(%r13)
	jne	.L1477
.LVL2485:
.LBB12216:
.LBB12217:
.LBB12218:
	.loc 2 709 0
	leaq	-448(%rbp), %rax
.LVL2486:
.LBE12218:
.LBE12217:
.LBB12221:
.LBB12222:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE12222:
.LBE12221:
.LBB12226:
.LBB12227:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm3
.LBE12227:
.LBE12226:
.LBE12216:
.LBB12241:
.LBB12242:
.LBB12243:
	.loc 2 353 0
	leaq	-256(%rbp), %rdi
.LBE12243:
.LBE12242:
.LBE12241:
.LBB12268:
.LBB12232:
.LBB12219:
	.loc 2 709 0
	addq	$8, %rax
.LVL2487:
.LBE12219:
.LBE12232:
.LBB12233:
.LBB12228:
	.loc 2 62 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
.LBE12228:
.LBE12233:
.LBB12234:
.LBB12220:
	.loc 2 709 0
	movq	%rax, -384(%rbp)
.LVL2488:
.LBE12220:
.LBE12234:
.LBB12235:
.LBB12223:
	.loc 2 738 0
	leaq	-448(%rbp), %rax
.LVL2489:
.LBE12223:
.LBE12235:
.LBB12236:
.LBB12229:
	.loc 2 62 0
	movq	$0, -416(%rbp)
.LBE12229:
.LBE12236:
.LBB12237:
.LBB12224:
	.loc 2 738 0
	movaps	%xmm0, -368(%rbp)
.LVL2490:
	addq	$80, %rax
.LVL2491:
.LBE12224:
.LBE12237:
.LBB12238:
.LBB12230:
	.loc 2 62 0
	movq	$0, -432(%rbp)
	.loc 2 63 0
	movq	$0, -424(%rbp)
	.loc 2 60 0
	movaps	%xmm3, -448(%rbp)
.LBE12230:
.LBE12238:
.LBE12268:
.LBB12269:
.LBB12247:
.LBB12244:
	.loc 2 353 0
	movl	$8, %ecx
	movl	$2, %esi
.LBE12244:
.LBE12247:
.LBE12269:
.LBB12270:
.LBB12271:
.LBB12272:
	.loc 2 738 0
	movaps	%xmm0, -272(%rbp)
.LBE12272:
.LBE12271:
.LBB12276:
.LBB12277:
	.loc 2 60 0
	movaps	%xmm3, -352(%rbp)
.LBE12277:
.LBE12276:
.LBE12270:
.LBB12293:
.LBB12248:
.LBB12249:
	.loc 2 738 0
	movaps	%xmm0, -176(%rbp)
.LBE12249:
.LBE12248:
.LBB12251:
.LBB12252:
	.loc 2 60 0
	movaps	%xmm3, -256(%rbp)
.LBE12252:
.LBE12251:
.LBE12293:
.LBB12294:
.LBB12239:
.LBB12225:
	.loc 2 738 0
	movq	%rax, -376(%rbp)
.LBE12225:
.LBE12239:
.LBE12294:
.LBB12295:
.LBB12281:
.LBB12282:
	.loc 2 709 0
	leaq	-352(%rbp), %rax
.LVL2492:
.LBE12282:
.LBE12281:
.LBE12295:
.LBB12296:
.LBB12240:
.LBB12231:
	.loc 2 64 0
	movq	$0, -392(%rbp)
.LVL2493:
.LBE12231:
.LBE12240:
.LBE12296:
.LBB12297:
.LBB12285:
.LBB12278:
	.loc 2 62 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
.LBE12278:
.LBE12285:
.LBB12286:
.LBB12283:
	.loc 2 709 0
	addq	$8, %rax
.LVL2494:
.LBE12283:
.LBE12286:
.LBB12287:
.LBB12279:
	.loc 2 62 0
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
.LBE12279:
.LBE12287:
.LBB12288:
.LBB12284:
	.loc 2 709 0
	movq	%rax, -288(%rbp)
.LVL2495:
.LBE12284:
.LBE12288:
.LBB12289:
.LBB12273:
	.loc 2 738 0
	leaq	-352(%rbp), %rax
.LVL2496:
.LBE12273:
.LBE12289:
.LBB12290:
.LBB12280:
	.loc 2 63 0
	movq	$0, -328(%rbp)
	.loc 2 64 0
	movq	$0, -296(%rbp)
.LVL2497:
.LBE12280:
.LBE12290:
.LBE12297:
.LBB12298:
.LBB12256:
.LBB12253:
	.loc 2 62 0
	movq	$0, -208(%rbp)
.LBE12253:
.LBE12256:
.LBE12298:
.LBB12299:
.LBB12291:
.LBB12274:
	.loc 2 738 0
	addq	$80, %rax
.LBE12274:
.LBE12291:
.LBE12299:
.LBB12300:
.LBB12257:
.LBB12254:
	.loc 2 62 0
	movq	$0, -216(%rbp)
	movq	$0, -224(%rbp)
.LBE12254:
.LBE12257:
.LBE12300:
.LBB12301:
.LBB12292:
.LBB12275:
	.loc 2 738 0
	movq	%rax, -280(%rbp)
.LBE12275:
.LBE12292:
.LBE12301:
.LBB12302:
.LBB12258:
.LBB12259:
	.loc 2 709 0
	leaq	-256(%rbp), %rax
.LBE12259:
.LBE12258:
.LBB12262:
.LBB12255:
	.loc 2 62 0
	movq	$0, -240(%rbp)
	.loc 2 63 0
	movq	$0, -232(%rbp)
	.loc 2 64 0
	movq	$0, -200(%rbp)
.LBE12255:
.LBE12262:
.LBB12263:
.LBB12260:
	.loc 2 709 0
	addq	$8, %rax
.LBE12260:
.LBE12263:
.LBB12264:
.LBB12245:
	.loc 2 352 0
	movl	%r12d, -480(%rbp)
.LBE12245:
.LBE12264:
.LBB12265:
.LBB12261:
	.loc 2 709 0
	movq	%rax, -192(%rbp)
.LVL2498:
.LBE12261:
.LBE12265:
.LBB12266:
.LBB12250:
	.loc 2 738 0
	leaq	-256(%rbp), %rax
	addq	$80, %rax
	movq	%rax, -184(%rbp)
.LBE12250:
.LBE12266:
.LBB12267:
.LBB12246:
	.loc 2 352 0
	movl	%edx, -476(%rbp)
	.loc 2 353 0
	leaq	-480(%rbp), %rdx
.LVL2499:
.LEHB42:
	call	_ZN2cv3Mat6createEiPKii
.LVL2500:
.LEHE42:
.LBE12246:
.LBE12267:
.LBE12302:
.LBB12303:
.LBB12304:
	.loc 2 285 0
	movq	-232(%rbp), %rax
	testq	%rax, %rax
	je	.L1343
	.loc 2 286 0
	lock addl	$1, (%rax)
.L1343:
.LVL2501:
.LBB12305:
.LBB12306:
	.loc 2 366 0
	movq	-424(%rbp), %rax
	testq	%rax, %rax
	je	.L1397
	lock subl	$1, (%rax)
	je	.L1478
.L1397:
.LBB12307:
	.loc 2 369 0
	movl	-444(%rbp), %edx
.LBE12307:
	.loc 2 368 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
.LVL2502:
.LBB12308:
	.loc 2 369 0
	testl	%edx, %edx
	jle	.L1479
	movq	-384(%rbp), %rdx
	xorl	%eax, %eax
.LVL2503:
	.p2align 4,,10
	.p2align 3
.L1346:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-444(%rbp), %ecx
	addl	$1, %eax
.LVL2504:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L1346
.LBE12308:
.LBE12306:
.LBE12305:
	.loc 2 288 0
	movl	-256(%rbp), %eax
.LVL2505:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB12312:
.LBB12309:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL2506:
.LBE12309:
.LBE12312:
	.loc 2 288 0
	movl	%eax, -448(%rbp)
	.loc 2 289 0
	jg	.L1347
.L1405:
	movl	-252(%rbp), %eax
	cmpl	$2, %eax
	jle	.L1480
.L1347:
	.loc 2 298 0
	leaq	-256(%rbp), %rsi
.LVL2507:
	leaq	-448(%rbp), %rdi
.LVL2508:
.LEHB43:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL2509:
.LEHE43:
.L1348:
	.loc 2 299 0
	movdqa	-240(%rbp), %xmm0
	.loc 2 303 0
	movq	-232(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -432(%rbp)
.LBE12304:
.LBE12303:
.LBB12319:
.LBB12320:
.LBB12321:
.LBB12322:
	.loc 2 366 0
	testq	%rax, %rax
.LBE12322:
.LBE12321:
.LBE12320:
.LBE12319:
.LBB12330:
.LBB12315:
	.loc 2 299 0
	movdqa	-224(%rbp), %xmm0
	movaps	%xmm0, -416(%rbp)
	movdqa	-208(%rbp), %xmm0
	movaps	%xmm0, -400(%rbp)
.LVL2510:
.LBE12315:
.LBE12330:
.LBB12331:
.LBB12329:
.LBB12327:
.LBB12325:
	.loc 2 366 0
	je	.L1398
	lock subl	$1, (%rax)
	jne	.L1398
	.loc 2 367 0
	leaq	-256(%rbp), %rdi
.LVL2511:
	call	_ZN2cv3Mat10deallocateEv
.LVL2512:
.L1398:
.LBB12323:
	.loc 2 369 0
	movl	-252(%rbp), %r8d
.LBE12323:
	.loc 2 368 0
	movq	$0, -208(%rbp)
	movq	$0, -216(%rbp)
	movq	$0, -224(%rbp)
	movq	$0, -240(%rbp)
.LVL2513:
.LBB12324:
	.loc 2 369 0
	testl	%r8d, %r8d
	jle	.L1354
	movq	-192(%rbp), %rdx
	xorl	%eax, %eax
.LVL2514:
	.p2align 4,,10
	.p2align 3
.L1355:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL2515:
	addq	$4, %rdx
	cmpl	%eax, -252(%rbp)
	jg	.L1355
.LVL2516:
.L1354:
.LBE12324:
.LBE12325:
.LBE12327:
	.loc 2 277 0
	movq	-184(%rbp), %rdi
	leaq	-256(%rbp), %rax
.LVL2517:
.LBB12328:
.LBB12326:
	.loc 2 371 0
	movq	$0, -232(%rbp)
.LVL2518:
.LBE12326:
.LBE12328:
	.loc 2 277 0
	addq	$80, %rax
.LVL2519:
	cmpq	%rax, %rdi
	je	.L1353
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL2520:
.L1353:
.LBE12329:
.LBE12331:
.LBB12332:
.LBB12333:
.LBB12334:
	.loc 2 709 0
	leaq	-160(%rbp), %rax
.LVL2521:
.LBE12334:
.LBE12333:
.LBB12337:
.LBB12338:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE12338:
.LBE12337:
.LBB12342:
.LBB12343:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm3
.LBE12343:
.LBE12342:
.LBB12346:
.LBB12347:
	.loc 2 353 0
	leaq	-464(%rbp), %rdx
.LBE12347:
.LBE12346:
.LBB12352:
.LBB12335:
	.loc 2 709 0
	addq	$8, %rax
.LVL2522:
.LBE12335:
.LBE12352:
.LBB12353:
.LBB12348:
	.loc 2 353 0
	leaq	-160(%rbp), %rdi
.LVL2523:
	movl	$13, %ecx
.LBE12348:
.LBE12353:
.LBB12354:
.LBB12336:
	.loc 2 709 0
	movq	%rax, -96(%rbp)
.LVL2524:
.LBE12336:
.LBE12354:
.LBB12355:
.LBB12339:
	.loc 2 738 0
	leaq	-160(%rbp), %rax
.LBE12339:
.LBE12355:
.LBB12356:
.LBB12349:
	.loc 2 353 0
	movl	$2, %esi
.LBE12349:
.LBE12356:
.LBB12357:
.LBB12340:
	.loc 2 738 0
	movaps	%xmm0, -80(%rbp)
.LVL2525:
	addq	$80, %rax
.LBE12340:
.LBE12357:
.LBB12358:
.LBB12344:
	.loc 2 62 0
	movq	$0, -112(%rbp)
	movq	$0, -120(%rbp)
	.loc 2 60 0
	movaps	%xmm3, -160(%rbp)
.LBE12344:
.LBE12358:
.LBB12359:
.LBB12341:
	.loc 2 738 0
	movq	%rax, -88(%rbp)
.LBE12341:
.LBE12359:
.LBB12360:
.LBB12350:
	.loc 2 352 0
	movl	-632(%rbp), %eax
.LBE12350:
.LBE12360:
.LBB12361:
.LBB12345:
	.loc 2 62 0
	movq	$0, -128(%rbp)
	movq	$0, -144(%rbp)
	.loc 2 63 0
	movq	$0, -136(%rbp)
	.loc 2 64 0
	movq	$0, -104(%rbp)
.LVL2526:
.LBE12345:
.LBE12361:
.LBB12362:
.LBB12351:
	.loc 2 352 0
	movl	%eax, -464(%rbp)
	movl	-616(%rbp), %eax
	movl	%eax, -460(%rbp)
.LEHB44:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL2527:
.LEHE44:
.LBE12351:
.LBE12362:
.LBE12332:
.LBB12363:
.LBB12364:
	.loc 2 285 0
	movq	-136(%rbp), %rax
	testq	%rax, %rax
	je	.L1356
	.loc 2 286 0
	lock addl	$1, (%rax)
.L1356:
.LVL2528:
.LBB12365:
.LBB12366:
	.loc 2 366 0
	movq	-328(%rbp), %rax
	testq	%rax, %rax
	je	.L1399
	lock subl	$1, (%rax)
	je	.L1481
.L1399:
.LBB12367:
	.loc 2 369 0
	movl	-348(%rbp), %eax
.LBE12367:
	.loc 2 368 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
.LVL2529:
.LBB12368:
	.loc 2 369 0
	testl	%eax, %eax
	jle	.L1482
	movq	-288(%rbp), %rdx
	xorl	%eax, %eax
.LVL2530:
	.p2align 4,,10
	.p2align 3
.L1359:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-348(%rbp), %ecx
	addl	$1, %eax
.LVL2531:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L1359
.LBE12368:
.LBE12366:
.LBE12365:
	.loc 2 288 0
	movl	-160(%rbp), %eax
.LVL2532:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB12372:
.LBB12369:
	.loc 2 371 0
	movq	$0, -328(%rbp)
.LVL2533:
.LBE12369:
.LBE12372:
	.loc 2 288 0
	movl	%eax, -352(%rbp)
	.loc 2 289 0
	jg	.L1360
.L1406:
	movl	-156(%rbp), %eax
	cmpl	$2, %eax
	jle	.L1483
.L1360:
	.loc 2 298 0
	leaq	-160(%rbp), %rsi
.LVL2534:
	leaq	-352(%rbp), %rdi
.LVL2535:
.LEHB45:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL2536:
.LEHE45:
.L1361:
	.loc 2 299 0
	movdqa	-144(%rbp), %xmm0
	.loc 2 303 0
	movq	-136(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -336(%rbp)
.LBE12364:
.LBE12363:
.LBB12379:
.LBB12380:
.LBB12381:
.LBB12382:
	.loc 2 366 0
	testq	%rax, %rax
.LBE12382:
.LBE12381:
.LBE12380:
.LBE12379:
.LBB12390:
.LBB12375:
	.loc 2 299 0
	movdqa	-128(%rbp), %xmm0
	movaps	%xmm0, -320(%rbp)
	movdqa	-112(%rbp), %xmm0
	movaps	%xmm0, -304(%rbp)
.LVL2537:
.LBE12375:
.LBE12390:
.LBB12391:
.LBB12389:
.LBB12387:
.LBB12385:
	.loc 2 366 0
	je	.L1402
	lock subl	$1, (%rax)
	jne	.L1402
	.loc 2 367 0
	leaq	-160(%rbp), %rdi
.LVL2538:
	call	_ZN2cv3Mat10deallocateEv
.LVL2539:
.L1402:
.LBB12383:
	.loc 2 369 0
	movl	-156(%rbp), %edi
.LBE12383:
	.loc 2 368 0
	movq	$0, -112(%rbp)
	movq	$0, -120(%rbp)
	movq	$0, -128(%rbp)
	movq	$0, -144(%rbp)
.LVL2540:
.LBB12384:
	.loc 2 369 0
	testl	%edi, %edi
	jle	.L1367
	movq	-96(%rbp), %rdx
	xorl	%eax, %eax
.LVL2541:
	.p2align 4,,10
	.p2align 3
.L1368:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL2542:
	addq	$4, %rdx
	cmpl	%eax, -156(%rbp)
	jg	.L1368
.LVL2543:
.L1367:
.LBE12384:
.LBE12385:
.LBE12387:
	.loc 2 277 0
	movq	-88(%rbp), %rdi
	leaq	-160(%rbp), %rax
.LVL2544:
.LBB12388:
.LBB12386:
	.loc 2 371 0
	movq	$0, -136(%rbp)
.LVL2545:
.LBE12386:
.LBE12388:
	.loc 2 277 0
	addq	$80, %rax
.LVL2546:
	cmpq	%rax, %rdi
	je	.L1366
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL2547:
.L1366:
.LBE12389:
.LBE12391:
.LBB12392:
	.loc 1 99 0
	testl	%r12d, %r12d
	jle	.L1374
	movslq	-624(%rbp), %r15
	movslq	%ebx, %rax
	movl	%ebx, -704(%rbp)
	movq	%rax, -648(%rbp)
.LVL2548:
	leaq	(%rax,%rax), %r14
	movq	%r13, %rbx
.LVL2549:
	leaq	(%r15,%r15), %rax
	xorl	%r15d, %r15d
	movq	%rax, %r13
.LVL2550:
	.p2align 4,,10
	.p2align 3
.L1372:
.LBB12393:
.LBB12394:
.LBB12395:
	.loc 2 430 0 discriminator 2
	movq	-376(%rbp), %rax
.LBE12395:
.LBE12394:
.LBB12396:
.LBB12397:
	.loc 5 53 0 discriminator 2
	movq	%r15, %rdi
	movq	%r15, %rsi
	movq	%r13, %rdx
	addq	$1, %r15
.LVL2551:
	imulq	(%rax), %rdi
.LBE12397:
.LBE12396:
.LBB12399:
.LBB12400:
	.loc 2 436 0 discriminator 2
	movq	72(%rbx), %rax
.LBE12400:
.LBE12399:
.LBB12401:
.LBB12398:
	.loc 5 53 0 discriminator 2
	imulq	(%rax), %rsi
.LVL2552:
	addq	%r14, %rdi
	addq	-432(%rbp), %rdi
	addq	16(%rbx), %rsi
	call	memcpy
.LVL2553:
.LBE12398:
.LBE12401:
.LBE12393:
	.loc 1 99 0 discriminator 2
	cmpl	%r15d, %r12d
	jg	.L1372
	movl	-640(%rbp), %esi
	movl	-704(%rbp), %ebx
.LVL2554:
	.loc 1 99 0 is_stmt 0
	xorl	%r9d, %r9d
	movl	%esi, %eax
	movslq	%esi, %r8
.LBE12392:
.LBB12402:
.LBB12403:
.LBB12404:
	.loc 1 110 0 is_stmt 1
	subq	-648(%rbp), %r8
	subl	%ebx, %eax
	cltq
	leaq	-2(%rax,%rax), %r11
	leal	-1(%rbx), %eax
	leaq	2(%rax,%rax), %r10
	.p2align 4,,10
	.p2align 3
.L1375:
.LVL2555:
.LBE12404:
.LBB12405:
.LBB12406:
	.loc 2 430 0
	movq	-376(%rbp), %rax
	movq	%r9, %rcx
	imulq	(%rax), %rcx
	movq	%rcx, %rax
	addq	-432(%rbp), %rax
.LVL2556:
.LBE12406:
.LBE12405:
.LBB12407:
	.loc 1 108 0
	testl	%ebx, %ebx
	leaq	(%rax,%r14), %rdi
	leaq	(%rax,%r11), %rsi
	leaq	(%rax,%r10), %rcx
	jle	.L1376
.LVL2557:
	.p2align 4,,10
	.p2align 3
.L1377:
	.loc 1 109 0 discriminator 2
	movzwl	(%rdi), %edx
	movw	%dx, (%rax)
	.loc 1 110 0 discriminator 2
	movzwl	(%rsi), %edx
	movw	%dx, (%rax,%r8,2)
	addq	$2, %rax
	.loc 1 108 0 discriminator 2
	cmpq	%rcx, %rax
	jne	.L1377
.L1376:
.LVL2558:
	addq	$1, %r9
.LVL2559:
.LBE12407:
.LBE12403:
	.loc 1 106 0
	cmpl	%r9d, %r12d
	jg	.L1375
.LVL2560:
.L1374:
.LBE12402:
	.loc 1 116 0
	testb	$15, -616(%rbp)
	jne	.L1484
.LBB12408:
	.loc 1 120 0
	movq	-664(%rbp), %rax
	pxor	%xmm0, %xmm0
	leaq	-592(%rbp), %rsi
	xorl	%ecx, %ecx
	movl	$4, %edx
	movl	$_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.1, %edi
	movl	%ebx, -520(%rbp)
	movl	%r12d, -512(%rbp)
	movq	%rax, -552(%rbp)
	movq	-656(%rbp), %rax
	movaps	%xmm0, -592(%rbp)
	movl	$2, -496(%rbp)
	movq	%rax, -560(%rbp)
	leaq	-600(%rbp), %rax
	movq	%rax, -576(%rbp)
	leaq	-596(%rbp), %rax
	movq	%rax, -568(%rbp)
	leaq	-448(%rbp), %rax
	movq	%rax, -544(%rbp)
	leaq	-352(%rbp), %rax
	movq	%rax, -536(%rbp)
	movq	-672(%rbp), %rax
	movq	%rax, -528(%rbp)
	movl	-676(%rbp), %eax
	movl	%eax, -516(%rbp)
	movl	-616(%rbp), %eax
	movl	%eax, -508(%rbp)
	movl	-624(%rbp), %eax
	movl	%eax, -504(%rbp)
	movl	-632(%rbp), %eax
	movl	%eax, -500(%rbp)
	call	GOMP_parallel
.LVL2561:
.LBE12408:
.LBB12409:
.LBB12410:
.LBB12411:
.LBB12412:
	.loc 2 366 0
	movq	-328(%rbp), %rax
	testq	%rax, %rax
	je	.L1403
	lock subl	$1, (%rax)
	jne	.L1403
	.loc 2 367 0
	leaq	-352(%rbp), %rdi
.LVL2562:
	call	_ZN2cv3Mat10deallocateEv
.LVL2563:
.L1403:
.LBB12413:
	.loc 2 369 0
	movl	-348(%rbp), %esi
.LBE12413:
	.loc 2 368 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
.LVL2564:
.LBB12414:
	.loc 2 369 0
	testl	%esi, %esi
	jle	.L1383
	movq	-288(%rbp), %rdx
	xorl	%eax, %eax
.LVL2565:
	.p2align 4,,10
	.p2align 3
.L1384:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL2566:
	addq	$4, %rdx
	cmpl	%eax, -348(%rbp)
	jg	.L1384
.LVL2567:
.L1383:
.LBE12414:
.LBE12412:
.LBE12411:
	.loc 2 277 0
	movq	-280(%rbp), %rdi
	leaq	-352(%rbp), %rax
.LVL2568:
.LBB12416:
.LBB12415:
	.loc 2 371 0
	movq	$0, -328(%rbp)
.LVL2569:
.LBE12415:
.LBE12416:
	.loc 2 277 0
	addq	$80, %rax
.LVL2570:
	cmpq	%rax, %rdi
	je	.L1382
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL2571:
.L1382:
.LBE12410:
.LBE12409:
.LBB12417:
.LBB12418:
.LBB12419:
.LBB12420:
	.loc 2 366 0
	movq	-424(%rbp), %rax
	testq	%rax, %rax
	je	.L1404
	lock subl	$1, (%rax)
	jne	.L1404
	.loc 2 367 0
	leaq	-448(%rbp), %rdi
.LVL2572:
	call	_ZN2cv3Mat10deallocateEv
.LVL2573:
.L1404:
.LBB12421:
	.loc 2 369 0
	movl	-444(%rbp), %ecx
.LBE12421:
	.loc 2 368 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
.LVL2574:
.LBB12422:
	.loc 2 369 0
	testl	%ecx, %ecx
	jle	.L1390
	movq	-384(%rbp), %rdx
	xorl	%eax, %eax
.LVL2575:
	.p2align 4,,10
	.p2align 3
.L1391:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL2576:
	addq	$4, %rdx
	cmpl	%eax, -444(%rbp)
	jg	.L1391
.LVL2577:
.L1390:
.LBE12422:
.LBE12420:
.LBE12419:
	.loc 2 277 0
	movq	-376(%rbp), %rdi
	leaq	-448(%rbp), %rax
.LVL2578:
.LBB12424:
.LBB12423:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL2579:
.LBE12423:
.LBE12424:
	.loc 2 277 0
	addq	$80, %rax
.LVL2580:
	cmpq	%rax, %rdi
	je	.L1302
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL2581:
.L1302:
.LBE12418:
.LBE12417:
	.loc 1 293 0
	movq	-56(%rbp), %rax
	xorq	%fs:40, %rax
	jne	.L1485
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
.LVL2582:
	ret
.LVL2583:
.L1483:
	.cfi_restore_state
.LBB12425:
.LBB12376:
	.loc 2 291 0
	movl	%eax, -348(%rbp)
	.loc 2 292 0
	movl	-152(%rbp), %eax
	movq	-88(%rbp), %rdx
	movl	%eax, -344(%rbp)
	.loc 2 293 0
	movl	-148(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -340(%rbp)
	movq	-280(%rbp), %rax
.LVL2584:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL2585:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L1361
.LVL2586:
.L1480:
.LBE12376:
.LBE12425:
.LBB12426:
.LBB12316:
	.loc 2 291 0
	movl	%eax, -444(%rbp)
	.loc 2 292 0
	movl	-248(%rbp), %eax
	movq	-184(%rbp), %rdx
	movl	%eax, -440(%rbp)
	.loc 2 293 0
	movl	-244(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -436(%rbp)
	movq	-376(%rbp), %rax
.LVL2587:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL2588:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L1348
.LVL2589:
.L1476:
	testl	%edx, %edx
	jne	.L1333
.LBE12316:
.LBE12426:
.LBB12427:
	.loc 1 72 0
	xorl	%ecx, %ecx
	jmp	.L1334
.LVL2590:
.L1473:
	testl	%edx, %edx
	jne	.L1318
.LBE12427:
.LBB12428:
	.loc 1 65 0
	xorl	%ecx, %ecx
	jmp	.L1319
.LVL2591:
.L1481:
.LBE12428:
.LBB12429:
.LBB12377:
.LBB12373:
.LBB12370:
	.loc 2 367 0
	leaq	-352(%rbp), %rdi
.LVL2592:
.LEHB46:
	call	_ZN2cv3Mat10deallocateEv
.LVL2593:
.LEHE46:
	jmp	.L1399
.LVL2594:
.L1478:
.LBE12370:
.LBE12373:
.LBE12377:
.LBE12429:
.LBB12430:
.LBB12317:
.LBB12313:
.LBB12310:
	leaq	-448(%rbp), %rdi
.LVL2595:
.LEHB47:
	call	_ZN2cv3Mat10deallocateEv
.LVL2596:
.LEHE47:
	jmp	.L1397
.LVL2597:
.L1482:
.LBE12310:
.LBE12313:
.LBE12317:
.LBE12430:
.LBB12431:
.LBB12378:
	.loc 2 288 0
	movl	-160(%rbp), %eax
.LBB12374:
.LBB12371:
	.loc 2 371 0
	movq	$0, -328(%rbp)
.LVL2598:
.LBE12371:
.LBE12374:
	.loc 2 288 0
	movl	%eax, -352(%rbp)
	jmp	.L1406
.LVL2599:
.L1479:
.LBE12378:
.LBE12431:
.LBB12432:
.LBB12318:
	movl	-256(%rbp), %eax
.LBB12314:
.LBB12311:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL2600:
.LBE12311:
.LBE12314:
	.loc 2 288 0
	movl	%eax, -448(%rbp)
	jmp	.L1405
.LVL2601:
.L1303:
.LBE12318:
.LBE12432:
	.loc 1 47 0 discriminator 3
	movl	$_ZZ12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, %ecx
	movl	$47, %edx
	movl	$.LC45, %esi
	movl	$.LC68, %edi
.LVL2602:
	call	__assert_fail
.LVL2603:
.L1419:
	movq	%rax, %rbx
.LVL2604:
	jmp	.L1392
.LVL2605:
.L1418:
	movq	%rax, %rbx
.LVL2606:
	jmp	.L1393
.LVL2607:
.L1477:
	.loc 1 83 0 discriminator 1
	movl	$_ZZ12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, %ecx
	movl	$83, %edx
.LVL2608:
	movl	$.LC45, %esi
	movl	$.LC70, %edi
	call	__assert_fail
.LVL2609:
.L1392:
	.loc 1 91 0 discriminator 2
	leaq	-256(%rbp), %rdi
.LVL2610:
	call	_ZN2cv3MatD1Ev
.LVL2611:
.L1393:
	.loc 1 85 0
	leaq	-352(%rbp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL2612:
	.loc 1 84 0
	leaq	-448(%rbp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL2613:
	movq	%rbx, %rdi
.LEHB48:
	call	_Unwind_Resume
.LVL2614:
.LEHE48:
.L1305:
	.loc 1 54 0 discriminator 3
	movl	$_ZZ12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, %ecx
.LVL2615:
	movl	$54, %edx
.LVL2616:
	movl	$.LC45, %esi
	movl	$.LC69, %edi
.LVL2617:
	call	__assert_fail
.LVL2618:
.L1484:
	.loc 1 116 0 discriminator 1
	movl	$_ZZ12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, %ecx
	movl	$116, %edx
	movl	$.LC45, %esi
	movl	$.LC71, %edi
	call	__assert_fail
.LVL2619:
.L1420:
	movq	%rax, %rbx
.LVL2620:
	jmp	.L1394
.LVL2621:
.L1485:
	.loc 1 293 0
	call	__stack_chk_fail
.LVL2622:
.L1394:
	.loc 1 92 0 discriminator 2
	leaq	-160(%rbp), %rdi
.LVL2623:
	call	_ZN2cv3MatD1Ev
.LVL2624:
	jmp	.L1393
	.cfi_endproc
.LFE11597:
	.section	.gcc_except_table
.LLSDA11597:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE11597-.LLSDACSB11597
.LLSDACSB11597:
	.uleb128 .LEHB42-.LFB11597
	.uleb128 .LEHE42-.LEHB42
	.uleb128 .L1418-.LFB11597
	.uleb128 0
	.uleb128 .LEHB43-.LFB11597
	.uleb128 .LEHE43-.LEHB43
	.uleb128 .L1419-.LFB11597
	.uleb128 0
	.uleb128 .LEHB44-.LFB11597
	.uleb128 .LEHE44-.LEHB44
	.uleb128 .L1418-.LFB11597
	.uleb128 0
	.uleb128 .LEHB45-.LFB11597
	.uleb128 .LEHE45-.LEHB45
	.uleb128 .L1420-.LFB11597
	.uleb128 0
	.uleb128 .LEHB46-.LFB11597
	.uleb128 .LEHE46-.LEHB46
	.uleb128 .L1420-.LFB11597
	.uleb128 0
	.uleb128 .LEHB47-.LFB11597
	.uleb128 .LEHE47-.LEHB47
	.uleb128 .L1419-.LFB11597
	.uleb128 0
	.uleb128 .LEHB48-.LFB11597
	.uleb128 .LEHE48-.LEHB48
	.uleb128 0
	.uleb128 0
.LLSDACSE11597:
	.section	.text._Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
	.size	_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd, .-_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd
	.section	.text.unlikely._Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LCOLDE73:
	.section	.text._Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.LHOTE73:
	.section	.text.unlikely._Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd,comdat
.LCOLDB74:
	.section	.text._Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd,comdat
.LHOTB74:
	.p2align 4,,15
	.weak	_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd
	.type	_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd, @function
_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd:
.LFB11598:
	.loc 1 40 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA11598
.LVL2625:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movapd	%xmm0, %xmm3
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$728, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	.loc 1 40 0
	movq	%rsi, -760(%rbp)
	movsd	.LC67(%rip), %xmm2
	movq	%fs:40, %rax
	movq	%rax, -56(%rbp)
	xorl	%eax, %eax
.LVL2626:
	mulsd	%xmm2, %xmm3
	mulsd	%xmm1, %xmm2
	cvttsd2si	%xmm3, %eax
	leal	1(%rax,%rax), %ecx
	movl	(%rdx), %eax
	cmpl	%eax, %ecx
	cmovg	%eax, %ecx
.LVL2627:
	cvttsd2si	%xmm2, %eax
	.loc 1 45 0
	movl	%ecx, -600(%rbp)
.LVL2628:
	leal	1(%rax,%rax), %esi
.LVL2629:
	movl	4(%rdx), %eax
	cmpl	%eax, %esi
	cmovle	%esi, %eax
.LVL2630:
	.loc 1 47 0
	movl	%ecx, %esi
	shrl	$31, %esi
	.loc 1 46 0
	movl	%eax, -596(%rbp)
	.loc 1 47 0
	leal	(%rcx,%rsi), %edx
.LVL2631:
	andl	$1, %edx
	subl	%esi, %edx
	cmpl	$1, %edx
	jne	.L1487
	.loc 1 47 0 is_stmt 0 discriminator 2
	movl	%eax, %esi
	shrl	$31, %esi
	leal	(%rax,%rsi), %edx
	andl	$1, %edx
	subl	%esi, %edx
	cmpl	$1, %edx
	jne	.L1487
	.loc 1 48 0 is_stmt 1
	movl	%ecx, %edx
	.loc 1 54 0
	pxor	%xmm7, %xmm7
	.loc 1 48 0
	shrl	$31, %edx
	addl	%edx, %ecx
	.loc 1 49 0
	movl	%eax, %edx
	shrl	$31, %edx
	.loc 1 48 0
	sarl	%ecx
	.loc 1 49 0
	addl	%edx, %eax
	.loc 1 48 0
	movl	%ecx, %r15d
.LVL2632:
	.loc 1 50 0
	leal	1(%rcx), %edx
	.loc 1 49 0
	sarl	%eax
	.loc 1 51 0
	leal	1(%rax), %ecx
.LVL2633:
	.loc 1 49 0
	movl	%eax, -752(%rbp)
.LVL2634:
	.loc 1 50 0
	movl	%edx, -600(%rbp)
	.loc 1 52 0
	movslq	%ecx, %rax
.LVL2635:
	.loc 1 51 0
	movl	%ecx, -596(%rbp)
.LVL2636:
	.loc 1 52 0
	leaq	18(,%rax,4), %rax
	andq	$-16, %rax
	subq	%rax, %rsp
	leaq	3(%rsp), %r13
	shrq	$2, %r13
	leaq	0(,%r13,4), %rax
	movq	%rax, -736(%rbp)
.LVL2637:
	.loc 1 53 0
	movslq	%edx, %rax
.LVL2638:
	leaq	18(,%rax,4), %rax
	andq	$-16, %rax
	subq	%rax, %rsp
	leaq	3(%rsp), %r12
	shrq	$2, %r12
	.loc 1 54 0
	ucomisd	%xmm7, %xmm0
	.loc 1 53 0
	leaq	0(,%r12,4), %rax
	movq	%rax, -744(%rbp)
.LVL2639:
	.loc 1 54 0
	jbe	.L1489
	.loc 1 54 0 is_stmt 0 discriminator 2
	ucomisd	%xmm7, %xmm1
	jbe	.L1489
	.loc 1 55 0 is_stmt 1
	movsd	.LC38(%rip), %xmm3
	movapd	%xmm0, %xmm4
	movsd	.LC15(%rip), %xmm2
.LBB12527:
	.loc 1 60 0
	testl	%ecx, %ecx
.LBE12527:
	.loc 1 55 0
	mulsd	%xmm3, %xmm4
	movq	%rdi, %rbx
	.loc 1 56 0
	mulsd	%xmm1, %xmm3
	movapd	%xmm2, %xmm6
	.loc 1 55 0
	movapd	%xmm2, %xmm5
	divsd	%xmm4, %xmm5
	.loc 1 56 0
	divsd	%xmm3, %xmm6
	.loc 1 57 0
	movapd	%xmm0, %xmm3
	addsd	%xmm0, %xmm3
	.loc 1 55 0
	movsd	%xmm5, -624(%rbp)
.LVL2640:
	.loc 1 57 0
	mulsd	%xmm3, %xmm0
.LVL2641:
	movapd	%xmm2, %xmm3
	.loc 1 56 0
	movsd	%xmm6, -632(%rbp)
.LVL2642:
	.loc 1 57 0
	divsd	%xmm0, %xmm3
	.loc 1 58 0
	movapd	%xmm1, %xmm0
	addsd	%xmm1, %xmm0
	mulsd	%xmm0, %xmm1
.LVL2643:
	.loc 1 57 0
	movsd	%xmm3, -640(%rbp)
.LVL2644:
	.loc 1 58 0
	divsd	%xmm1, %xmm2
	movsd	%xmm2, -648(%rbp)
.LVL2645:
.LBB12528:
	.loc 1 60 0
	jle	.L1492
	xorl	%r14d, %r14d
	pxor	%xmm1, %xmm1
	movl	%r14d, %ebx
	movq	%rdi, -656(%rbp)
	movq	-736(%rbp), %r14
	jmp	.L1498
.LVL2646:
	.p2align 4,,10
	.p2align 3
.L1690:
	.loc 1 62 0 discriminator 1
	addsd	%xmm0, %xmm0
	.loc 1 60 0 discriminator 1
	movl	-596(%rbp), %eax
	addl	$1, %ebx
.LVL2647:
	cmpl	%ebx, %eax
	.loc 1 62 0 discriminator 1
	addsd	%xmm0, %xmm1
.LVL2648:
	.loc 1 60 0 discriminator 1
	jle	.L1689
.LVL2649:
.L1498:
	.loc 1 61 0
	movl	%ebx, %eax
	pxor	%xmm0, %xmm0
	negl	%eax
	movsd	%xmm1, -616(%rbp)
.LVL2650:
	imull	%ebx, %eax
	cvtsi2sd	%eax, %xmm0
	mulsd	-648(%rbp), %xmm0
	call	exp
.LVL2651:
	mulsd	-632(%rbp), %xmm0
	movslq	%ebx, %rax
	.loc 1 62 0
	testl	%ebx, %ebx
	movsd	-616(%rbp), %xmm1
	.loc 1 61 0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, (%r14,%rax,4)
	.loc 1 62 0
	cvtss2sd	%xmm0, %xmm0
	jg	.L1690
.LVL2652:
	.loc 1 60 0
	movl	-596(%rbp), %eax
	addl	$1, %ebx
.LVL2653:
	.loc 1 63 0
	addsd	%xmm0, %xmm1
.LVL2654:
	.loc 1 60 0
	cmpl	%ebx, %eax
	jg	.L1498
.L1689:
.LBE12528:
.LBB12529:
	.loc 1 65 0
	testl	%eax, %eax
	movq	-656(%rbp), %rbx
.LVL2655:
	jle	.L1505
	movq	-736(%rbp), %rdx
	andl	$15, %edx
	shrq	$2, %rdx
	negq	%rdx
	andl	$3, %edx
	cmpl	%eax, %edx
	cmova	%eax, %edx
	cmpl	$4, %eax
	jg	.L1691
	movl	%eax, %edx
.L1502:
	pxor	%xmm0, %xmm0
	cmpl	$1, %edx
	movl	$1, %ecx
	pxor	%xmm4, %xmm4
	cvtss2sd	0(,%r13,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm4
	movss	%xmm4, 0(,%r13,4)
.LVL2656:
	je	.L1504
	pxor	%xmm0, %xmm0
	cmpl	$2, %edx
	movl	$2, %ecx
	pxor	%xmm5, %xmm5
	cvtss2sd	4(,%r13,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm5
	movss	%xmm5, 4(,%r13,4)
.LVL2657:
	je	.L1504
	pxor	%xmm0, %xmm0
	cmpl	$3, %edx
	movl	$3, %ecx
	pxor	%xmm6, %xmm6
	cvtss2sd	8(,%r13,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm6
	movss	%xmm6, 8(,%r13,4)
.LVL2658:
	je	.L1504
	pxor	%xmm0, %xmm0
	movl	$4, %ecx
	pxor	%xmm7, %xmm7
	cvtss2sd	12(,%r13,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm7
	movss	%xmm7, 12(,%r13,4)
.LVL2659:
.L1504:
	cmpl	%edx, %eax
	je	.L1505
.L1503:
	movl	%eax, %r8d
	leal	-1(%rax), %edi
	movl	%edx, %r10d
	subl	%edx, %r8d
	leal	-4(%r8), %esi
	subl	%edx, %edi
	shrl	$2, %esi
	addl	$1, %esi
	cmpl	$2, %edi
	leal	0(,%rsi,4), %r9d
	jbe	.L1506
	movq	-736(%rbp), %rdx
	movddup	%xmm1, %xmm3
	xorl	%edi, %edi
	leaq	(%rdx,%r10,4), %r10
	xorl	%edx, %edx
.L1508:
	.loc 1 65 0 is_stmt 0 discriminator 2
	movaps	(%r10,%rdx), %xmm2
	addl	$1, %edi
	movhps	%xmm2, -672(%rbp)
	cvtps2pd	%xmm2, %xmm0
	cvtps2pd	-672(%rbp), %xmm2
	divpd	%xmm3, %xmm0
	divpd	%xmm3, %xmm2
	cvtpd2ps	%xmm0, %xmm0
	cvtpd2ps	%xmm2, %xmm2
	movlhps	%xmm2, %xmm0
	movaps	%xmm0, (%r10,%rdx)
	addq	$16, %rdx
	cmpl	%esi, %edi
	jb	.L1508
	addl	%r9d, %ecx
	cmpl	%r9d, %r8d
	je	.L1505
.L1506:
.LVL2660:
	movq	-736(%rbp), %rsi
	movslq	%ecx, %rdx
	.loc 1 65 0
	pxor	%xmm0, %xmm0
	pxor	%xmm4, %xmm4
	leaq	(%rsi,%rdx,4), %rdx
	cvtss2sd	(%rdx), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm4
	movss	%xmm4, (%rdx)
	leal	1(%rcx), %edx
.LVL2661:
	cmpl	%edx, %eax
	jle	.L1505
	movslq	%edx, %rdx
	pxor	%xmm0, %xmm0
	leaq	(%rsi,%rdx,4), %rdx
.LVL2662:
	pxor	%xmm5, %xmm5
	addl	$2, %ecx
.LVL2663:
	cvtss2sd	(%rdx), %xmm0
	cmpl	%ecx, %eax
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm5
	movss	%xmm5, (%rdx)
.LVL2664:
	jle	.L1505
	movslq	%ecx, %rcx
	pxor	%xmm0, %xmm0
	leaq	(%rsi,%rcx,4), %rax
	pxor	%xmm7, %xmm7
	cvtss2sd	(%rax), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm7
	movss	%xmm7, (%rax)
.L1505:
	movl	-600(%rbp), %edx
.LVL2665:
.L1492:
.LBE12529:
.LBB12530:
	.loc 1 67 0 is_stmt 1 discriminator 1
	xorl	%r13d, %r13d
	testl	%edx, %edx
	movq	-744(%rbp), %r14
	pxor	%xmm1, %xmm1
	jg	.L1654
	jmp	.L1511
.LVL2666:
	.p2align 4,,10
	.p2align 3
.L1693:
	.loc 1 69 0 discriminator 1
	addsd	%xmm0, %xmm0
	.loc 1 67 0 discriminator 1
	movl	-600(%rbp), %eax
	addl	$1, %r13d
.LVL2667:
	cmpl	%r13d, %eax
	.loc 1 69 0 discriminator 1
	addsd	%xmm0, %xmm1
.LVL2668:
	.loc 1 67 0 discriminator 1
	jle	.L1692
.LVL2669:
.L1654:
	.loc 1 68 0
	movl	%r13d, %eax
	pxor	%xmm0, %xmm0
	negl	%eax
	movsd	%xmm1, -616(%rbp)
.LVL2670:
	imull	%r13d, %eax
	cvtsi2sd	%eax, %xmm0
	mulsd	-640(%rbp), %xmm0
	call	exp
.LVL2671:
	mulsd	-624(%rbp), %xmm0
	movslq	%r13d, %rax
	.loc 1 69 0
	testl	%r13d, %r13d
	movsd	-616(%rbp), %xmm1
	.loc 1 68 0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, (%r14,%rax,4)
	.loc 1 69 0
	cvtss2sd	%xmm0, %xmm0
	jg	.L1693
.LVL2672:
	.loc 1 67 0
	movl	-600(%rbp), %eax
	addl	$1, %r13d
.LVL2673:
	.loc 1 70 0
	addsd	%xmm0, %xmm1
.LVL2674:
	.loc 1 67 0
	cmpl	%r13d, %eax
	jg	.L1654
.L1692:
.LVL2675:
.LBE12530:
.LBB12531:
	.loc 1 72 0 discriminator 3
	testl	%eax, %eax
	jle	.L1511
	movq	-744(%rbp), %rdx
	andl	$15, %edx
	shrq	$2, %rdx
	negq	%rdx
	andl	$3, %edx
	cmpl	%eax, %edx
	cmova	%eax, %edx
	cmpl	$4, %eax
	jg	.L1694
	.loc 1 72 0 is_stmt 0
	movl	%eax, %edx
.L1517:
	pxor	%xmm0, %xmm0
	cmpl	$1, %edx
	movl	$1, %ecx
	pxor	%xmm4, %xmm4
	cvtss2sd	0(,%r12,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm4
	movss	%xmm4, 0(,%r12,4)
.LVL2676:
	je	.L1519
	pxor	%xmm0, %xmm0
	cmpl	$2, %edx
	movl	$2, %ecx
	pxor	%xmm5, %xmm5
	cvtss2sd	4(,%r12,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm5
	movss	%xmm5, 4(,%r12,4)
.LVL2677:
	je	.L1519
	pxor	%xmm0, %xmm0
	cmpl	$3, %edx
	movl	$3, %ecx
	pxor	%xmm6, %xmm6
	cvtss2sd	8(,%r12,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm6
	movss	%xmm6, 8(,%r12,4)
.LVL2678:
	je	.L1519
	pxor	%xmm0, %xmm0
	movl	$4, %ecx
	pxor	%xmm3, %xmm3
	cvtss2sd	12(,%r12,4), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm3
	movss	%xmm3, 12(,%r12,4)
.LVL2679:
.L1519:
	cmpl	%edx, %eax
	je	.L1511
.L1518:
	movl	%eax, %r8d
	leal	-1(%rax), %edi
	movl	%edx, %r10d
	subl	%edx, %r8d
	leal	-4(%r8), %esi
	subl	%edx, %edi
	shrl	$2, %esi
	addl	$1, %esi
	cmpl	$2, %edi
	leal	0(,%rsi,4), %r9d
	jbe	.L1521
	movq	-744(%rbp), %rdx
	movddup	%xmm1, %xmm3
	xorl	%edi, %edi
	leaq	(%rdx,%r10,4), %r10
	xorl	%edx, %edx
.L1523:
	.loc 1 72 0 discriminator 2
	movaps	(%r10,%rdx), %xmm2
	addl	$1, %edi
	movhps	%xmm2, -688(%rbp)
	cvtps2pd	%xmm2, %xmm0
	cvtps2pd	-688(%rbp), %xmm2
	divpd	%xmm3, %xmm0
	divpd	%xmm3, %xmm2
	cvtpd2ps	%xmm0, %xmm0
	cvtpd2ps	%xmm2, %xmm2
	movlhps	%xmm2, %xmm0
	movaps	%xmm0, (%r10,%rdx)
	addq	$16, %rdx
	cmpl	%esi, %edi
	jb	.L1523
	addl	%r9d, %ecx
	cmpl	%r9d, %r8d
	je	.L1511
.L1521:
.LVL2680:
	movq	-744(%rbp), %rsi
	movslq	%ecx, %rdx
	.loc 1 72 0
	pxor	%xmm0, %xmm0
	pxor	%xmm3, %xmm3
	leaq	(%rsi,%rdx,4), %rdx
	cvtss2sd	(%rdx), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm3
	movss	%xmm3, (%rdx)
	leal	1(%rcx), %edx
.LVL2681:
	cmpl	%edx, %eax
	jle	.L1511
	movslq	%edx, %rdx
	pxor	%xmm0, %xmm0
	leaq	(%rsi,%rdx,4), %rdx
.LVL2682:
	pxor	%xmm7, %xmm7
	addl	$2, %ecx
.LVL2683:
	cvtss2sd	(%rdx), %xmm0
	cmpl	%ecx, %eax
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm7
	movss	%xmm7, (%rdx)
.LVL2684:
	jle	.L1511
	movslq	%ecx, %rcx
	pxor	%xmm0, %xmm0
	leaq	(%rsi,%rcx,4), %rax
	pxor	%xmm6, %xmm6
	cvtss2sd	(%rax), %xmm0
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm6
	movss	%xmm6, (%rax)
.LVL2685:
.L1511:
.LBE12531:
	.loc 1 74 0 is_stmt 1
	movl	8(%rbx), %eax
	movl	%eax, -640(%rbp)
.LVL2686:
	.loc 1 75 0
	movl	12(%rbx), %eax
.LVL2687:
	.loc 1 78 0
	testb	$15, %al
	.loc 1 75 0
	movl	%eax, -724(%rbp)
.LVL2688:
	.loc 1 78 0
	je	.L1680
	movl	%eax, %ecx
	.loc 1 78 0 is_stmt 0 discriminator 1
	addl	$15, %eax
.LVL2689:
	testl	%ecx, %ecx
	cmovns	%ecx, %eax
	andl	$-16, %eax
	addl	$16, %eax
.LVL2690:
.L1680:
	movl	%eax, %ecx
.LVL2691:
	movl	%eax, -728(%rbp)
.LVL2692:
	.loc 1 80 0 is_stmt 1 discriminator 1
	leal	(%r15,%r15), %eax
	leal	(%rcx,%rax), %edx
.LVL2693:
	.loc 1 81 0 discriminator 1
	movl	-724(%rbp), %ecx
	leal	(%rcx,%rax), %r14d
.LVL2694:
	.loc 1 82 0 discriminator 1
	movl	-640(%rbp), %eax
	movl	-752(%rbp), %ecx
	leal	(%rax,%rcx,2), %eax
	movl	%eax, -748(%rbp)
.LVL2695:
	.loc 1 83 0 discriminator 1
	movq	-760(%rbp), %rax
.LVL2696:
	movq	8(%rax), %rax
	cmpq	%rax, 8(%rbx)
	jne	.L1695
.LVL2697:
.LBB12532:
.LBB12533:
.LBB12534:
	.loc 2 709 0
	leaq	-448(%rbp), %rax
.LVL2698:
.LBE12534:
.LBE12533:
.LBB12537:
.LBB12538:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE12538:
.LBE12537:
.LBB12542:
.LBB12543:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm3
.LBE12543:
.LBE12542:
.LBE12532:
.LBB12557:
.LBB12558:
.LBB12559:
	.loc 2 353 0
	leaq	-256(%rbp), %rdi
.LBE12559:
.LBE12558:
.LBE12557:
.LBB12580:
.LBB12548:
.LBB12535:
	.loc 2 709 0
	addq	$8, %rax
.LVL2699:
.LBE12535:
.LBE12548:
.LBB12549:
.LBB12544:
	.loc 2 62 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
.LBE12544:
.LBE12549:
.LBB12550:
.LBB12536:
	.loc 2 709 0
	movq	%rax, -384(%rbp)
.LVL2700:
.LBE12536:
.LBE12550:
.LBB12551:
.LBB12539:
	.loc 2 738 0
	leaq	-448(%rbp), %rax
.LVL2701:
.LBE12539:
.LBE12551:
.LBB12552:
.LBB12545:
	.loc 2 62 0
	movq	$0, -416(%rbp)
.LBE12545:
.LBE12552:
.LBB12553:
.LBB12540:
	.loc 2 738 0
	movaps	%xmm0, -368(%rbp)
.LVL2702:
	addq	$80, %rax
.LVL2703:
.LBE12540:
.LBE12553:
.LBB12554:
.LBB12546:
	.loc 2 62 0
	movq	$0, -432(%rbp)
	.loc 2 63 0
	movq	$0, -424(%rbp)
	.loc 2 60 0
	movaps	%xmm3, -448(%rbp)
.LBE12546:
.LBE12554:
.LBE12580:
.LBB12581:
.LBB12562:
.LBB12560:
	.loc 2 353 0
	xorl	%ecx, %ecx
	movl	$2, %esi
.LBE12560:
.LBE12562:
.LBE12581:
.LBB12582:
.LBB12583:
.LBB12584:
	.loc 2 738 0
	movaps	%xmm0, -272(%rbp)
.LBE12584:
.LBE12583:
.LBB12588:
.LBB12589:
	.loc 2 60 0
	movaps	%xmm3, -352(%rbp)
.LBE12589:
.LBE12588:
.LBE12582:
.LBB12605:
.LBB12563:
.LBB12564:
	.loc 2 738 0
	movaps	%xmm0, -176(%rbp)
.LBE12564:
.LBE12563:
.LBB12566:
.LBB12567:
	.loc 2 60 0
	movaps	%xmm3, -256(%rbp)
.LBE12567:
.LBE12566:
.LBE12605:
.LBB12606:
.LBB12555:
.LBB12541:
	.loc 2 738 0
	movq	%rax, -376(%rbp)
.LBE12541:
.LBE12555:
.LBE12606:
.LBB12607:
.LBB12593:
.LBB12594:
	.loc 2 709 0
	leaq	-352(%rbp), %rax
.LVL2704:
.LBE12594:
.LBE12593:
.LBE12607:
.LBB12608:
.LBB12556:
.LBB12547:
	.loc 2 64 0
	movq	$0, -392(%rbp)
.LVL2705:
.LBE12547:
.LBE12556:
.LBE12608:
.LBB12609:
.LBB12597:
.LBB12590:
	.loc 2 62 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
.LBE12590:
.LBE12597:
.LBB12598:
.LBB12595:
	.loc 2 709 0
	addq	$8, %rax
.LVL2706:
.LBE12595:
.LBE12598:
.LBB12599:
.LBB12591:
	.loc 2 62 0
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
.LBE12591:
.LBE12599:
.LBB12600:
.LBB12596:
	.loc 2 709 0
	movq	%rax, -288(%rbp)
.LVL2707:
.LBE12596:
.LBE12600:
.LBB12601:
.LBB12585:
	.loc 2 738 0
	leaq	-352(%rbp), %rax
.LVL2708:
.LBE12585:
.LBE12601:
.LBB12602:
.LBB12592:
	.loc 2 63 0
	movq	$0, -328(%rbp)
	.loc 2 64 0
	movq	$0, -296(%rbp)
.LVL2709:
.LBE12592:
.LBE12602:
.LBE12609:
.LBB12610:
.LBB12571:
.LBB12568:
	.loc 2 62 0
	movq	$0, -208(%rbp)
.LBE12568:
.LBE12571:
.LBE12610:
.LBB12611:
.LBB12603:
.LBB12586:
	.loc 2 738 0
	addq	$80, %rax
.LBE12586:
.LBE12603:
.LBE12611:
.LBB12612:
.LBB12572:
.LBB12569:
	.loc 2 62 0
	movq	$0, -216(%rbp)
	movq	$0, -224(%rbp)
.LBE12569:
.LBE12572:
.LBE12612:
.LBB12613:
.LBB12604:
.LBB12587:
	.loc 2 738 0
	movq	%rax, -280(%rbp)
.LBE12587:
.LBE12604:
.LBE12613:
.LBB12614:
.LBB12573:
.LBB12574:
	.loc 2 709 0
	leaq	-256(%rbp), %rax
.LBE12574:
.LBE12573:
.LBB12576:
.LBB12570:
	.loc 2 62 0
	movq	$0, -240(%rbp)
	.loc 2 63 0
	movq	$0, -232(%rbp)
	.loc 2 64 0
	movq	$0, -200(%rbp)
.LBE12570:
.LBE12576:
.LBB12577:
.LBB12575:
	.loc 2 709 0
	addq	$8, %rax
	movq	%rax, -192(%rbp)
.LVL2710:
.LBE12575:
.LBE12577:
.LBB12578:
.LBB12565:
	.loc 2 738 0
	leaq	-256(%rbp), %rax
	addq	$80, %rax
	movq	%rax, -184(%rbp)
.LBE12565:
.LBE12578:
.LBB12579:
.LBB12561:
	.loc 2 352 0
	movl	-640(%rbp), %eax
	movl	%eax, -480(%rbp)
	movl	%edx, -476(%rbp)
	.loc 2 353 0
	leaq	-480(%rbp), %rdx
.LVL2711:
.LEHB49:
	call	_ZN2cv3Mat6createEiPKii
.LVL2712:
.LEHE49:
.LBE12561:
.LBE12579:
.LBE12614:
.LBB12615:
.LBB12616:
	.loc 2 285 0
	movq	-232(%rbp), %rax
	testq	%rax, %rax
	je	.L1527
	.loc 2 286 0
	lock addl	$1, (%rax)
.L1527:
.LVL2713:
.LBB12617:
.LBB12618:
	.loc 2 366 0
	movq	-424(%rbp), %rax
	testq	%rax, %rax
	je	.L1590
	lock subl	$1, (%rax)
	je	.L1696
.L1590:
.LBB12619:
	.loc 2 369 0
	movl	-444(%rbp), %edx
.LBE12619:
	.loc 2 368 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
.LVL2714:
.LBB12620:
	.loc 2 369 0
	testl	%edx, %edx
	jle	.L1697
	movq	-384(%rbp), %rdx
	xorl	%eax, %eax
.LVL2715:
	.p2align 4,,10
	.p2align 3
.L1530:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-444(%rbp), %ecx
	addl	$1, %eax
.LVL2716:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L1530
.LBE12620:
.LBE12618:
.LBE12617:
	.loc 2 288 0
	movl	-256(%rbp), %eax
.LVL2717:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB12624:
.LBB12621:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL2718:
.LBE12621:
.LBE12624:
	.loc 2 288 0
	movl	%eax, -448(%rbp)
	.loc 2 289 0
	jg	.L1531
.L1598:
	movl	-252(%rbp), %eax
	cmpl	$2, %eax
	jle	.L1698
.L1531:
	.loc 2 298 0
	leaq	-256(%rbp), %rsi
.LVL2719:
	leaq	-448(%rbp), %rdi
.LVL2720:
.LEHB50:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL2721:
.LEHE50:
.L1532:
	.loc 2 299 0
	movdqa	-240(%rbp), %xmm0
	.loc 2 303 0
	movq	-232(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -432(%rbp)
.LBE12616:
.LBE12615:
.LBB12631:
.LBB12632:
.LBB12633:
.LBB12634:
	.loc 2 366 0
	testq	%rax, %rax
.LBE12634:
.LBE12633:
.LBE12632:
.LBE12631:
.LBB12642:
.LBB12627:
	.loc 2 299 0
	movdqa	-224(%rbp), %xmm0
	movaps	%xmm0, -416(%rbp)
	movdqa	-208(%rbp), %xmm0
	movaps	%xmm0, -400(%rbp)
.LVL2722:
.LBE12627:
.LBE12642:
.LBB12643:
.LBB12641:
.LBB12639:
.LBB12637:
	.loc 2 366 0
	je	.L1591
	lock subl	$1, (%rax)
	jne	.L1591
	.loc 2 367 0
	leaq	-256(%rbp), %rdi
.LVL2723:
	call	_ZN2cv3Mat10deallocateEv
.LVL2724:
.L1591:
.LBB12635:
	.loc 2 369 0
	movl	-252(%rbp), %r9d
.LBE12635:
	.loc 2 368 0
	movq	$0, -208(%rbp)
	movq	$0, -216(%rbp)
	movq	$0, -224(%rbp)
	movq	$0, -240(%rbp)
.LVL2725:
.LBB12636:
	.loc 2 369 0
	testl	%r9d, %r9d
	jle	.L1538
	movq	-192(%rbp), %rdx
	xorl	%eax, %eax
.LVL2726:
	.p2align 4,,10
	.p2align 3
.L1539:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL2727:
	addq	$4, %rdx
	cmpl	%eax, -252(%rbp)
	jg	.L1539
.LVL2728:
.L1538:
.LBE12636:
.LBE12637:
.LBE12639:
	.loc 2 277 0
	movq	-184(%rbp), %rdi
	leaq	-256(%rbp), %rax
.LVL2729:
.LBB12640:
.LBB12638:
	.loc 2 371 0
	movq	$0, -232(%rbp)
.LVL2730:
.LBE12638:
.LBE12640:
	.loc 2 277 0
	addq	$80, %rax
.LVL2731:
	cmpq	%rax, %rdi
	je	.L1537
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL2732:
.L1537:
.LBE12641:
.LBE12643:
.LBB12644:
.LBB12645:
.LBB12646:
	.loc 2 709 0
	leaq	-160(%rbp), %rax
.LVL2733:
.LBE12646:
.LBE12645:
.LBB12649:
.LBB12650:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE12650:
.LBE12649:
.LBB12654:
.LBB12655:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm3
.LBE12655:
.LBE12654:
.LBB12658:
.LBB12659:
	.loc 2 353 0
	leaq	-464(%rbp), %rdx
.LBE12659:
.LBE12658:
.LBB12664:
.LBB12647:
	.loc 2 709 0
	addq	$8, %rax
.LVL2734:
.LBE12647:
.LBE12664:
.LBB12665:
.LBB12660:
	.loc 2 353 0
	leaq	-160(%rbp), %rdi
.LVL2735:
	movl	$5, %ecx
.LBE12660:
.LBE12665:
.LBB12666:
.LBB12648:
	.loc 2 709 0
	movq	%rax, -96(%rbp)
.LVL2736:
.LBE12648:
.LBE12666:
.LBB12667:
.LBB12651:
	.loc 2 738 0
	leaq	-160(%rbp), %rax
.LBE12651:
.LBE12667:
.LBB12668:
.LBB12661:
	.loc 2 353 0
	movl	$2, %esi
.LBE12661:
.LBE12668:
.LBB12669:
.LBB12652:
	.loc 2 738 0
	movaps	%xmm0, -80(%rbp)
.LVL2737:
	addq	$80, %rax
.LBE12652:
.LBE12669:
.LBB12670:
.LBB12656:
	.loc 2 62 0
	movq	$0, -112(%rbp)
	movq	$0, -120(%rbp)
	.loc 2 60 0
	movaps	%xmm3, -160(%rbp)
.LBE12656:
.LBE12670:
.LBB12671:
.LBB12653:
	.loc 2 738 0
	movq	%rax, -88(%rbp)
.LBE12653:
.LBE12671:
.LBB12672:
.LBB12662:
	.loc 2 352 0
	movl	-748(%rbp), %eax
.LBE12662:
.LBE12672:
.LBB12673:
.LBB12657:
	.loc 2 62 0
	movq	$0, -128(%rbp)
	movq	$0, -144(%rbp)
	.loc 2 63 0
	movq	$0, -136(%rbp)
	.loc 2 64 0
	movq	$0, -104(%rbp)
.LVL2738:
.LBE12657:
.LBE12673:
.LBB12674:
.LBB12663:
	.loc 2 352 0
	movl	%eax, -464(%rbp)
	movl	-728(%rbp), %eax
	movl	%eax, -460(%rbp)
.LEHB51:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL2739:
.LEHE51:
.LBE12663:
.LBE12674:
.LBE12644:
.LBB12675:
.LBB12676:
	.loc 2 285 0
	movq	-136(%rbp), %rax
	testq	%rax, %rax
	je	.L1540
	.loc 2 286 0
	lock addl	$1, (%rax)
.L1540:
.LVL2740:
.LBB12677:
.LBB12678:
	.loc 2 366 0
	movq	-328(%rbp), %rax
	testq	%rax, %rax
	je	.L1592
	lock subl	$1, (%rax)
	je	.L1699
.L1592:
.LBB12679:
	.loc 2 369 0
	movl	-348(%rbp), %eax
.LBE12679:
	.loc 2 368 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
.LVL2741:
.LBB12680:
	.loc 2 369 0
	testl	%eax, %eax
	jle	.L1700
	movq	-288(%rbp), %rdx
	xorl	%eax, %eax
.LVL2742:
	.p2align 4,,10
	.p2align 3
.L1543:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-348(%rbp), %ecx
	addl	$1, %eax
.LVL2743:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L1543
.LBE12680:
.LBE12678:
.LBE12677:
	.loc 2 288 0
	movl	-160(%rbp), %eax
.LVL2744:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB12684:
.LBB12681:
	.loc 2 371 0
	movq	$0, -328(%rbp)
.LVL2745:
.LBE12681:
.LBE12684:
	.loc 2 288 0
	movl	%eax, -352(%rbp)
	.loc 2 289 0
	jg	.L1544
.L1599:
	movl	-156(%rbp), %eax
	cmpl	$2, %eax
	jle	.L1701
.L1544:
	.loc 2 298 0
	leaq	-160(%rbp), %rsi
.LVL2746:
	leaq	-352(%rbp), %rdi
.LVL2747:
.LEHB52:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL2748:
.LEHE52:
.L1545:
	.loc 2 299 0
	movdqa	-144(%rbp), %xmm0
	.loc 2 303 0
	movq	-136(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -336(%rbp)
.LBE12676:
.LBE12675:
.LBB12691:
.LBB12692:
.LBB12693:
.LBB12694:
	.loc 2 366 0
	testq	%rax, %rax
.LBE12694:
.LBE12693:
.LBE12692:
.LBE12691:
.LBB12702:
.LBB12687:
	.loc 2 299 0
	movdqa	-128(%rbp), %xmm0
	movaps	%xmm0, -320(%rbp)
	movdqa	-112(%rbp), %xmm0
	movaps	%xmm0, -304(%rbp)
.LVL2749:
.LBE12687:
.LBE12702:
.LBB12703:
.LBB12701:
.LBB12699:
.LBB12697:
	.loc 2 366 0
	je	.L1595
	lock subl	$1, (%rax)
	jne	.L1595
	.loc 2 367 0
	leaq	-160(%rbp), %rdi
.LVL2750:
	call	_ZN2cv3Mat10deallocateEv
.LVL2751:
.L1595:
.LBB12695:
	.loc 2 369 0
	movl	-156(%rbp), %r8d
.LBE12695:
	.loc 2 368 0
	movq	$0, -112(%rbp)
	movq	$0, -120(%rbp)
	movq	$0, -128(%rbp)
	movq	$0, -144(%rbp)
.LVL2752:
.LBB12696:
	.loc 2 369 0
	testl	%r8d, %r8d
	jle	.L1551
	movq	-96(%rbp), %rdx
	xorl	%eax, %eax
.LVL2753:
	.p2align 4,,10
	.p2align 3
.L1552:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL2754:
	addq	$4, %rdx
	cmpl	%eax, -156(%rbp)
	jg	.L1552
.LVL2755:
.L1551:
.LBE12696:
.LBE12697:
.LBE12699:
	.loc 2 277 0
	movq	-88(%rbp), %rdi
	leaq	-160(%rbp), %rax
.LVL2756:
.LBB12700:
.LBB12698:
	.loc 2 371 0
	movq	$0, -136(%rbp)
.LVL2757:
.LBE12698:
.LBE12700:
	.loc 2 277 0
	addq	$80, %rax
.LVL2758:
	cmpq	%rax, %rdi
	je	.L1550
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL2759:
.L1550:
.LBE12701:
.LBE12703:
.LBB12704:
	.loc 1 99 0
	movl	-640(%rbp), %edi
	movslq	%r15d, %rax
	xorl	%r13d, %r13d
	movq	%rax, -672(%rbp)
	movslq	-724(%rbp), %r12
	testl	%edi, %edi
	jle	.L1558
	movl	%r14d, -616(%rbp)
	movl	%r15d, -624(%rbp)
.LVL2760:
	movq	%r13, %r14
.LVL2761:
	movq	%r12, %r13
	movslq	%r15d, %r12
	movq	%rbx, %r15
.LVL2762:
	movl	-640(%rbp), %ebx
.LVL2763:
	.p2align 4,,10
	.p2align 3
.L1650:
.LBB12705:
.LBB12706:
.LBB12707:
	.loc 2 430 0 discriminator 2
	movq	-376(%rbp), %rax
.LBE12707:
.LBE12706:
.LBB12708:
.LBB12709:
	.loc 5 53 0 discriminator 2
	movq	%r14, %rdi
	movq	%r14, %rsi
	movq	%r13, %rdx
	addq	$1, %r14
.LVL2764:
	imulq	(%rax), %rdi
.LBE12709:
.LBE12708:
.LBB12711:
.LBB12712:
	.loc 2 436 0 discriminator 2
	movq	72(%r15), %rax
.LBE12712:
.LBE12711:
.LBB12713:
.LBB12710:
	.loc 5 53 0 discriminator 2
	imulq	(%rax), %rsi
.LVL2765:
	addq	%r12, %rdi
	addq	-432(%rbp), %rdi
	addq	16(%r15), %rsi
	call	memcpy
.LVL2766:
.LBE12710:
.LBE12713:
.LBE12705:
	.loc 1 99 0 discriminator 2
	cmpl	%r14d, %ebx
	jg	.L1650
	movl	-624(%rbp), %r15d
.LVL2767:
	movl	-724(%rbp), %eax
	.loc 1 99 0 is_stmt 0
	xorl	%r10d, %r10d
	movl	-616(%rbp), %r14d
	leal	(%rax,%r15,2), %eax
	movslq	%eax, %rdx
	subl	%r15d, %eax
	subq	-672(%rbp), %rdx
	cltq
	movq	%rax, %rbx
	movq	%rax, -688(%rbp)
	movl	%r14d, %eax
	subl	%r15d, %eax
	cltq
	movq	%rdx, -624(%rbp)
.LVL2768:
	movq	%rax, -704(%rbp)
	subq	$1, %rax
	movq	%rax, -656(%rbp)
	movq	%rbx, %rax
.LBE12704:
.LBB12714:
.LBB12715:
.LBB12716:
	.loc 1 110 0 is_stmt 1
	movq	%r10, %rbx
	subq	$1, %rax
	movq	%rax, -696(%rbp)
	leal	1(%r14), %eax
	subl	%r15d, %eax
	cltq
	movq	%rax, -712(%rbp)
	leal	2(%r14), %eax
	subl	%r15d, %eax
	cltq
	movq	%rax, -720(%rbp)
.LVL2769:
	.p2align 4,,10
	.p2align 3
.L1559:
	movq	-432(%rbp), %rax
.LBE12716:
.LBB12717:
.LBB12718:
	.loc 2 430 0
	movq	%rbx, %rsi
	movq	%rax, %rcx
	movq	%rax, -648(%rbp)
.LVL2770:
	movq	-376(%rbp), %rax
	imulq	(%rax), %rsi
	movq	%rcx, %rax
	addq	%rsi, %rax
.LVL2771:
.LBE12718:
.LBE12717:
.LBB12719:
	.loc 1 108 0
	testl	%r15d, %r15d
	jle	.L1568
	movq	-672(%rbp), %r13
	movq	-624(%rbp), %r9
	movl	%r15d, %r8d
	movq	-656(%rbp), %rcx
	movq	-696(%rbp), %r11
	movq	-688(%rbp), %r10
	movq	%r9, %rdi
	movq	%r13, %r12
	leaq	(%rax,%r13), %rdx
	leaq	1(%rsi,%r13), %r13
	addq	%r8, %rdi
	addq	%rsi, %r9
	addq	%rsi, %rdi
	addq	%rsi, %r10
	addq	%rsi, %r11
	addq	%rsi, %r12
	addq	%rsi, %r8
	addq	%rax, %rcx
	cmpq	%r9, %r13
	movq	%r13, -632(%rbp)
	movq	%r9, -616(%rbp)
	setle	%r13b
	cmpq	%rdi, %r12
	setge	%r9b
	orl	%r13d, %r9d
	cmpq	%r10, -616(%rbp)
	setge	%r13b
	cmpq	%rdi, %r11
	setge	%dil
	orl	%r13d, %edi
	andl	%edi, %r9d
	cmpl	$20, %r15d
	seta	%dil
	andl	%edi, %r9d
	movq	-624(%rbp), %rdi
	leaq	16(%rsi,%rdi), %rdi
	cmpq	%rdi, %rsi
	leaq	16(%rsi), %rdi
	setge	%r13b
	cmpq	-616(%rbp), %rdi
	setle	%dil
	orl	%r13d, %edi
	andl	%r9d, %edi
	cmpq	%r11, %r8
	setle	%r9b
	cmpq	%r10, %rsi
	setge	%r10b
	orl	%r10d, %r9d
	testb	%r9b, %dil
	je	.L1560
	cmpq	%r12, %r8
	setle	%dil
	cmpq	-632(%rbp), %rsi
	setge	%r8b
	orb	%r8b, %dil
	je	.L1560
	movq	%rax, %r8
	negq	%r8
	andl	$15, %r8d
	cmpl	%r15d, %r8d
	cmova	%r15d, %r8d
	xorl	%edi, %edi
	testl	%r8d, %r8d
	je	.L1561
	.loc 1 109 0
	movzbl	(%rdx), %edi
	.loc 1 110 0
	movq	-704(%rbp), %r11
	cmpl	$1, %r8d
	.loc 1 109 0
	movb	%dil, (%rax)
	.loc 1 110 0
	movzbl	(%rcx), %edi
	movb	%dil, (%rax,%r11)
.LVL2772:
	je	.L1612
	.loc 1 109 0
	movzbl	(%rdx), %edi
	.loc 1 110 0
	movq	-712(%rbp), %r11
	cmpl	$2, %r8d
	.loc 1 109 0
	movb	%dil, 1(%rax)
	.loc 1 110 0
	movzbl	(%rcx), %edi
	movb	%dil, (%rax,%r11)
.LVL2773:
	je	.L1613
	.loc 1 109 0
	movzbl	(%rdx), %edi
	.loc 1 110 0
	movq	-720(%rbp), %r11
	cmpl	$3, %r8d
	.loc 1 109 0
	movb	%dil, 2(%rax)
	.loc 1 110 0
	movzbl	(%rcx), %edi
	movb	%dil, (%rax,%r11)
.LVL2774:
	je	.L1614
	.loc 1 109 0
	movzbl	(%rdx), %edi
	movb	%dil, 3(%rax)
	.loc 1 110 0
	leal	3(%r14), %edi
	movzbl	(%rcx), %r9d
	subl	%r15d, %edi
	cmpl	$4, %r8d
	movslq	%edi, %rdi
	movb	%r9b, (%rax,%rdi)
.LVL2775:
	je	.L1615
	.loc 1 109 0
	movzbl	(%rdx), %edi
	movb	%dil, 4(%rax)
	.loc 1 110 0
	leal	4(%r14), %edi
	movzbl	(%rcx), %r9d
	subl	%r15d, %edi
	cmpl	$5, %r8d
	movslq	%edi, %rdi
	movb	%r9b, (%rax,%rdi)
.LVL2776:
	je	.L1616
	.loc 1 109 0
	movzbl	(%rdx), %edi
	movb	%dil, 5(%rax)
	.loc 1 110 0
	leal	5(%r14), %edi
	movzbl	(%rcx), %r9d
	subl	%r15d, %edi
	cmpl	$6, %r8d
	movslq	%edi, %rdi
	movb	%r9b, (%rax,%rdi)
.LVL2777:
	je	.L1617
	.loc 1 109 0
	movzbl	(%rdx), %edi
	movb	%dil, 6(%rax)
	.loc 1 110 0
	leal	6(%r14), %edi
	movzbl	(%rcx), %r9d
	subl	%r15d, %edi
	cmpl	$7, %r8d
	movslq	%edi, %rdi
	movb	%r9b, (%rax,%rdi)
.LVL2778:
	je	.L1618
	.loc 1 109 0
	movzbl	(%rdx), %edi
	movb	%dil, 7(%rax)
	.loc 1 110 0
	leal	7(%r14), %edi
	movzbl	(%rcx), %r9d
	subl	%r15d, %edi
	cmpl	$8, %r8d
	movslq	%edi, %rdi
	movb	%r9b, (%rax,%rdi)
.LVL2779:
	je	.L1619
	.loc 1 109 0
	movzbl	(%rdx), %edi
	movb	%dil, 8(%rax)
	.loc 1 110 0
	leal	8(%r14), %edi
	movzbl	(%rcx), %r9d
	subl	%r15d, %edi
	cmpl	$9, %r8d
	movslq	%edi, %rdi
	movb	%r9b, (%rax,%rdi)
.LVL2780:
	je	.L1620
	.loc 1 109 0
	movzbl	(%rdx), %edi
	movb	%dil, 9(%rax)
	.loc 1 110 0
	leal	9(%r14), %edi
	movzbl	(%rcx), %r9d
	subl	%r15d, %edi
	cmpl	$10, %r8d
	movslq	%edi, %rdi
	movb	%r9b, (%rax,%rdi)
.LVL2781:
	je	.L1621
	.loc 1 109 0
	movzbl	(%rdx), %edi
	movb	%dil, 10(%rax)
	.loc 1 110 0
	leal	10(%r14), %edi
	movzbl	(%rcx), %r9d
	subl	%r15d, %edi
	cmpl	$11, %r8d
	movslq	%edi, %rdi
	movb	%r9b, (%rax,%rdi)
.LVL2782:
	je	.L1622
	.loc 1 109 0
	movzbl	(%rdx), %edi
	movb	%dil, 11(%rax)
	.loc 1 110 0
	leal	11(%r14), %edi
	movzbl	(%rcx), %r9d
	subl	%r15d, %edi
	cmpl	$12, %r8d
	movslq	%edi, %rdi
	movb	%r9b, (%rax,%rdi)
.LVL2783:
	je	.L1623
	.loc 1 109 0
	movzbl	(%rdx), %edi
	movb	%dil, 12(%rax)
	.loc 1 110 0
	leal	12(%r14), %edi
	movzbl	(%rcx), %r9d
	subl	%r15d, %edi
	cmpl	$13, %r8d
	movslq	%edi, %rdi
	movb	%r9b, (%rax,%rdi)
.LVL2784:
	je	.L1624
	.loc 1 109 0
	movzbl	(%rdx), %edi
	movb	%dil, 13(%rax)
	.loc 1 110 0
	leal	13(%r14), %edi
	movzbl	(%rcx), %r9d
	subl	%r15d, %edi
	cmpl	$14, %r8d
	movslq	%edi, %rdi
	movb	%r9b, (%rax,%rdi)
.LVL2785:
	je	.L1625
	.loc 1 109 0
	movzbl	(%rdx), %edi
	movb	%dil, 14(%rax)
	.loc 1 110 0
	leal	14(%r14), %edi
	movzbl	(%rcx), %r9d
	subl	%r15d, %edi
	movslq	%edi, %rdi
	movb	%r9b, (%rax,%rdi)
.LVL2786:
	.loc 1 108 0
	movl	$15, %edi
.LVL2787:
.L1561:
	movl	%r15d, %r11d
	leal	-1(%r15), %r10d
	movl	%r8d, %r12d
	subl	%r8d, %r11d
	leal	-16(%r11), %r9d
	subl	%r8d, %r10d
	shrl	$4, %r9d
	addl	$1, %r9d
	movl	%r9d, %r13d
	sall	$4, %r13d
	cmpl	$14, %r10d
	jbe	.L1563
	movzbl	(%rdx), %r8d
	pxor	%xmm2, %xmm2
	addq	%r12, %rsi
	addq	-616(%rbp), %r12
	movq	-648(%rbp), %r10
	movl	%r8d, -632(%rbp)
	movzbl	(%rcx), %r8d
	addq	%r10, %rsi
	movd	-632(%rbp), %xmm0
	addq	%r10, %r12
	xorl	%r10d, %r10d
	movdqa	%xmm0, %xmm1
	movl	%r8d, -632(%rbp)
	xorl	%r8d, %r8d
	movd	-632(%rbp), %xmm0
	pshufb	%xmm2, %xmm1
	pshufb	%xmm2, %xmm0
.L1565:
	addl	$1, %r10d
	.loc 1 109 0 discriminator 2
	movaps	%xmm1, (%rsi,%r8)
	.loc 1 110 0 discriminator 2
	movups	%xmm0, (%r12,%r8)
	addq	$16, %r8
	cmpl	%r10d, %r9d
	ja	.L1565
	addl	%r13d, %edi
	cmpl	%r11d, %r13d
	je	.L1568
.L1563:
.LVL2788:
	.loc 1 109 0
	movzbl	(%rdx), %r8d
	movslq	%edi, %rsi
	movb	%r8b, (%rax,%rsi)
	.loc 1 110 0
	leal	(%r14,%rdi), %esi
	movzbl	(%rcx), %r8d
	subl	%r15d, %esi
	movslq	%esi, %rsi
	movb	%r8b, (%rax,%rsi)
	.loc 1 108 0
	leal	1(%rdi), %esi
.LVL2789:
	cmpl	%esi, %r15d
	jle	.L1568
	.loc 1 109 0
	movzbl	(%rdx), %r9d
	movslq	%esi, %r8
	.loc 1 110 0
	addl	%r14d, %esi
.LVL2790:
	subl	%r15d, %esi
	movslq	%esi, %rsi
	.loc 1 109 0
	movb	%r9b, (%rax,%r8)
	.loc 1 110 0
	movzbl	(%rcx), %r8d
.LVL2791:
	movb	%r8b, (%rax,%rsi)
	.loc 1 108 0
	leal	2(%rdi), %esi
.LVL2792:
	cmpl	%esi, %r15d
	jle	.L1568
	.loc 1 109 0
	movzbl	(%rdx), %r9d
	movslq	%esi, %r8
	.loc 1 110 0
	addl	%r14d, %esi
.LVL2793:
	subl	%r15d, %esi
	movslq	%esi, %rsi
	.loc 1 109 0
	movb	%r9b, (%rax,%r8)
	.loc 1 110 0
	movzbl	(%rcx), %r8d
.LVL2794:
	movb	%r8b, (%rax,%rsi)
	.loc 1 108 0
	leal	3(%rdi), %esi
.LVL2795:
	cmpl	%esi, %r15d
	jle	.L1568
	.loc 1 109 0
	movzbl	(%rdx), %r9d
	movslq	%esi, %r8
	.loc 1 110 0
	addl	%r14d, %esi
.LVL2796:
	subl	%r15d, %esi
	movslq	%esi, %rsi
	.loc 1 109 0
	movb	%r9b, (%rax,%r8)
	.loc 1 110 0
	movzbl	(%rcx), %r8d
.LVL2797:
	movb	%r8b, (%rax,%rsi)
	.loc 1 108 0
	leal	4(%rdi), %esi
.LVL2798:
	cmpl	%esi, %r15d
	jle	.L1568
	.loc 1 109 0
	movzbl	(%rdx), %r9d
	movslq	%esi, %r8
	.loc 1 110 0
	addl	%r14d, %esi
.LVL2799:
	subl	%r15d, %esi
	movslq	%esi, %rsi
	.loc 1 109 0
	movb	%r9b, (%rax,%r8)
	.loc 1 110 0
	movzbl	(%rcx), %r8d
.LVL2800:
	movb	%r8b, (%rax,%rsi)
	.loc 1 108 0
	leal	5(%rdi), %esi
.LVL2801:
	cmpl	%esi, %r15d
	jle	.L1568
	.loc 1 109 0
	movzbl	(%rdx), %r9d
	movslq	%esi, %r8
	.loc 1 110 0
	addl	%r14d, %esi
.LVL2802:
	subl	%r15d, %esi
	movslq	%esi, %rsi
	.loc 1 109 0
	movb	%r9b, (%rax,%r8)
	.loc 1 110 0
	movzbl	(%rcx), %r8d
.LVL2803:
	movb	%r8b, (%rax,%rsi)
	.loc 1 108 0
	leal	6(%rdi), %esi
.LVL2804:
	cmpl	%esi, %r15d
	jle	.L1568
	.loc 1 109 0
	movzbl	(%rdx), %r9d
	movslq	%esi, %r8
	.loc 1 110 0
	addl	%r14d, %esi
.LVL2805:
	subl	%r15d, %esi
	movslq	%esi, %rsi
	.loc 1 109 0
	movb	%r9b, (%rax,%r8)
	.loc 1 110 0
	movzbl	(%rcx), %r8d
.LVL2806:
	movb	%r8b, (%rax,%rsi)
	.loc 1 108 0
	leal	7(%rdi), %esi
.LVL2807:
	cmpl	%esi, %r15d
	jle	.L1568
	.loc 1 109 0
	movzbl	(%rdx), %r9d
	movslq	%esi, %r8
	.loc 1 110 0
	addl	%r14d, %esi
.LVL2808:
	subl	%r15d, %esi
	movslq	%esi, %rsi
	.loc 1 109 0
	movb	%r9b, (%rax,%r8)
	.loc 1 110 0
	movzbl	(%rcx), %r8d
.LVL2809:
	movb	%r8b, (%rax,%rsi)
	.loc 1 108 0
	leal	8(%rdi), %esi
.LVL2810:
	cmpl	%esi, %r15d
	jle	.L1568
	.loc 1 109 0
	movzbl	(%rdx), %r9d
	movslq	%esi, %r8
	.loc 1 110 0
	addl	%r14d, %esi
.LVL2811:
	subl	%r15d, %esi
	movslq	%esi, %rsi
	.loc 1 109 0
	movb	%r9b, (%rax,%r8)
	.loc 1 110 0
	movzbl	(%rcx), %r8d
.LVL2812:
	movb	%r8b, (%rax,%rsi)
	.loc 1 108 0
	leal	9(%rdi), %esi
.LVL2813:
	cmpl	%esi, %r15d
	jle	.L1568
	.loc 1 109 0
	movzbl	(%rdx), %r9d
	movslq	%esi, %r8
	.loc 1 110 0
	addl	%r14d, %esi
.LVL2814:
	subl	%r15d, %esi
	movslq	%esi, %rsi
	.loc 1 109 0
	movb	%r9b, (%rax,%r8)
	.loc 1 110 0
	movzbl	(%rcx), %r8d
.LVL2815:
	movb	%r8b, (%rax,%rsi)
	.loc 1 108 0
	leal	10(%rdi), %esi
.LVL2816:
	cmpl	%esi, %r15d
	jle	.L1568
	.loc 1 109 0
	movzbl	(%rdx), %r9d
	movslq	%esi, %r8
	.loc 1 110 0
	addl	%r14d, %esi
.LVL2817:
	subl	%r15d, %esi
	movslq	%esi, %rsi
	.loc 1 109 0
	movb	%r9b, (%rax,%r8)
	.loc 1 110 0
	movzbl	(%rcx), %r8d
.LVL2818:
	movb	%r8b, (%rax,%rsi)
	.loc 1 108 0
	leal	11(%rdi), %esi
.LVL2819:
	cmpl	%esi, %r15d
	jle	.L1568
	.loc 1 109 0
	movzbl	(%rdx), %r9d
	movslq	%esi, %r8
	.loc 1 110 0
	addl	%r14d, %esi
.LVL2820:
	subl	%r15d, %esi
	movslq	%esi, %rsi
	.loc 1 109 0
	movb	%r9b, (%rax,%r8)
	.loc 1 110 0
	movzbl	(%rcx), %r8d
.LVL2821:
	movb	%r8b, (%rax,%rsi)
	.loc 1 108 0
	leal	12(%rdi), %esi
.LVL2822:
	cmpl	%esi, %r15d
	jle	.L1568
	.loc 1 109 0
	movzbl	(%rdx), %r9d
	movslq	%esi, %r8
	.loc 1 110 0
	addl	%r14d, %esi
.LVL2823:
	subl	%r15d, %esi
	movslq	%esi, %rsi
	.loc 1 109 0
	movb	%r9b, (%rax,%r8)
	.loc 1 110 0
	movzbl	(%rcx), %r8d
.LVL2824:
	movb	%r8b, (%rax,%rsi)
	.loc 1 108 0
	leal	13(%rdi), %esi
.LVL2825:
	cmpl	%esi, %r15d
	jle	.L1568
	.loc 1 109 0
	movzbl	(%rdx), %r9d
	movslq	%esi, %r8
	.loc 1 110 0
	addl	%r14d, %esi
.LVL2826:
	subl	%r15d, %esi
	.loc 1 108 0
	addl	$14, %edi
	.loc 1 110 0
	movslq	%esi, %rsi
	.loc 1 108 0
	cmpl	%edi, %r15d
	.loc 1 109 0
	movb	%r9b, (%rax,%r8)
	.loc 1 110 0
	movzbl	(%rcx), %r8d
.LVL2827:
	movb	%r8b, (%rax,%rsi)
.LVL2828:
	.loc 1 108 0
	jle	.L1568
	.loc 1 109 0
	movzbl	(%rdx), %esi
	movslq	%edi, %rdx
	.loc 1 110 0
	addl	%r14d, %edi
	subl	%r15d, %edi
	.loc 1 109 0
	movb	%sil, (%rax,%rdx)
	.loc 1 110 0
	movzbl	(%rcx), %edx
	movslq	%edi, %rcx
	movb	%dl, (%rax,%rcx)
.L1568:
.LVL2829:
	addq	$1, %rbx
.LVL2830:
.LBE12719:
.LBE12715:
	.loc 1 106 0
	cmpl	%ebx, -640(%rbp)
	jg	.L1559
.LVL2831:
.L1558:
.LBE12714:
	.loc 1 116 0
	testb	$15, -728(%rbp)
	jne	.L1702
.LBB12724:
	.loc 1 120 0
	movq	-744(%rbp), %rax
	pxor	%xmm0, %xmm0
	leaq	-592(%rbp), %rsi
	xorl	%ecx, %ecx
	movl	$4, %edx
	movl	$_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd._omp_fn.2, %edi
	movl	%r15d, -520(%rbp)
	movl	$1, -496(%rbp)
	movq	%rax, -552(%rbp)
	movq	-736(%rbp), %rax
	movaps	%xmm0, -592(%rbp)
	movq	%rax, -560(%rbp)
	leaq	-600(%rbp), %rax
	movq	%rax, -576(%rbp)
	leaq	-596(%rbp), %rax
	movq	%rax, -568(%rbp)
	leaq	-448(%rbp), %rax
	movq	%rax, -544(%rbp)
	leaq	-352(%rbp), %rax
	movq	%rax, -536(%rbp)
	movq	-760(%rbp), %rax
	movq	%rax, -528(%rbp)
	movl	-752(%rbp), %eax
	movl	%eax, -516(%rbp)
	movl	-640(%rbp), %eax
	movl	%eax, -512(%rbp)
	movl	-728(%rbp), %eax
	movl	%eax, -508(%rbp)
	movl	-724(%rbp), %eax
	movl	%eax, -504(%rbp)
	movl	-748(%rbp), %eax
	movl	%eax, -500(%rbp)
	call	GOMP_parallel
.LVL2832:
.LBE12724:
.LBB12725:
.LBB12726:
.LBB12727:
.LBB12728:
	.loc 2 366 0
	movq	-328(%rbp), %rax
	testq	%rax, %rax
	je	.L1596
	lock subl	$1, (%rax)
	jne	.L1596
	.loc 2 367 0
	leaq	-352(%rbp), %rdi
.LVL2833:
	call	_ZN2cv3Mat10deallocateEv
.LVL2834:
.L1596:
.LBB12729:
	.loc 2 369 0
	movl	-348(%rbp), %esi
.LBE12729:
	.loc 2 368 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
.LVL2835:
.LBB12730:
	.loc 2 369 0
	testl	%esi, %esi
	jle	.L1576
	movq	-288(%rbp), %rdx
	xorl	%eax, %eax
.LVL2836:
	.p2align 4,,10
	.p2align 3
.L1577:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL2837:
	addq	$4, %rdx
	cmpl	%eax, -348(%rbp)
	jg	.L1577
.LVL2838:
.L1576:
.LBE12730:
.LBE12728:
.LBE12727:
	.loc 2 277 0
	movq	-280(%rbp), %rdi
	leaq	-352(%rbp), %rax
.LVL2839:
.LBB12732:
.LBB12731:
	.loc 2 371 0
	movq	$0, -328(%rbp)
.LVL2840:
.LBE12731:
.LBE12732:
	.loc 2 277 0
	addq	$80, %rax
.LVL2841:
	cmpq	%rax, %rdi
	je	.L1575
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL2842:
.L1575:
.LBE12726:
.LBE12725:
.LBB12733:
.LBB12734:
.LBB12735:
.LBB12736:
	.loc 2 366 0
	movq	-424(%rbp), %rax
	testq	%rax, %rax
	je	.L1597
	lock subl	$1, (%rax)
	jne	.L1597
	.loc 2 367 0
	leaq	-448(%rbp), %rdi
.LVL2843:
	call	_ZN2cv3Mat10deallocateEv
.LVL2844:
.L1597:
.LBB12737:
	.loc 2 369 0
	movl	-444(%rbp), %ecx
.LBE12737:
	.loc 2 368 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
.LVL2845:
.LBB12738:
	.loc 2 369 0
	testl	%ecx, %ecx
	jle	.L1583
	movq	-384(%rbp), %rdx
	xorl	%eax, %eax
.LVL2846:
	.p2align 4,,10
	.p2align 3
.L1584:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL2847:
	addq	$4, %rdx
	cmpl	%eax, -444(%rbp)
	jg	.L1584
.LVL2848:
.L1583:
.LBE12738:
.LBE12736:
.LBE12735:
	.loc 2 277 0
	movq	-376(%rbp), %rdi
	leaq	-448(%rbp), %rax
.LVL2849:
.LBB12740:
.LBB12739:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL2850:
.LBE12739:
.LBE12740:
	.loc 2 277 0
	addq	$80, %rax
.LVL2851:
	cmpq	%rax, %rdi
	je	.L1486
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL2852:
.L1486:
.LBE12734:
.LBE12733:
	.loc 1 293 0
	movq	-56(%rbp), %rax
	xorq	%fs:40, %rax
	jne	.L1703
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
.LVL2853:
	ret
.LVL2854:
	.p2align 4,,10
	.p2align 3
.L1560:
	.cfi_restore_state
	leal	-1(%r15), %esi
.LBB12741:
.LBB12722:
.LBB12720:
	.loc 1 110 0
	movq	-672(%rbp), %rdi
	movslq	%r14d, %r8
	leaq	1(%rax,%rsi), %r9
	negq	%rdi
.LVL2855:
	.p2align 4,,10
	.p2align 3
.L1570:
	.loc 1 109 0
	movzbl	(%rdx), %esi
	movb	%sil, (%rax)
	.loc 1 110 0
	movzbl	(%rcx), %r10d
	leaq	(%r8,%rax), %rsi
	addq	$1, %rax
	.loc 1 108 0
	cmpq	%r9, %rax
	.loc 1 110 0
	movb	%r10b, (%rsi,%rdi)
	.loc 1 108 0
	jne	.L1570
	jmp	.L1568
.LVL2856:
	.p2align 4,,10
	.p2align 3
.L1701:
.LBE12720:
.LBE12722:
.LBE12741:
.LBB12742:
.LBB12688:
	.loc 2 291 0
	movl	%eax, -348(%rbp)
	.loc 2 292 0
	movl	-152(%rbp), %eax
	movq	-88(%rbp), %rdx
	movl	%eax, -344(%rbp)
	.loc 2 293 0
	movl	-148(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -340(%rbp)
	movq	-280(%rbp), %rax
.LVL2857:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL2858:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L1545
.LVL2859:
	.p2align 4,,10
	.p2align 3
.L1698:
.LBE12688:
.LBE12742:
.LBB12743:
.LBB12628:
	.loc 2 291 0
	movl	%eax, -444(%rbp)
	.loc 2 292 0
	movl	-248(%rbp), %eax
	movq	-184(%rbp), %rdx
	movl	%eax, -440(%rbp)
	.loc 2 293 0
	movl	-244(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -436(%rbp)
	movq	-376(%rbp), %rax
.LVL2860:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL2861:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L1532
.LVL2862:
	.p2align 4,,10
	.p2align 3
.L1625:
.LBE12628:
.LBE12743:
.LBB12744:
.LBB12723:
.LBB12721:
	.loc 1 108 0
	movl	$14, %edi
	jmp	.L1561
.LVL2863:
	.p2align 4,,10
	.p2align 3
.L1624:
	movl	$13, %edi
	jmp	.L1561
.LVL2864:
	.p2align 4,,10
	.p2align 3
.L1623:
	movl	$12, %edi
	jmp	.L1561
.LVL2865:
	.p2align 4,,10
	.p2align 3
.L1622:
	movl	$11, %edi
	jmp	.L1561
.LVL2866:
	.p2align 4,,10
	.p2align 3
.L1621:
	movl	$10, %edi
	jmp	.L1561
.LVL2867:
	.p2align 4,,10
	.p2align 3
.L1620:
	movl	$9, %edi
	jmp	.L1561
.LVL2868:
	.p2align 4,,10
	.p2align 3
.L1619:
	movl	$8, %edi
	jmp	.L1561
.LVL2869:
	.p2align 4,,10
	.p2align 3
.L1618:
	movl	$7, %edi
	jmp	.L1561
.LVL2870:
	.p2align 4,,10
	.p2align 3
.L1617:
	movl	$6, %edi
	jmp	.L1561
.LVL2871:
	.p2align 4,,10
	.p2align 3
.L1616:
	movl	$5, %edi
	jmp	.L1561
.LVL2872:
	.p2align 4,,10
	.p2align 3
.L1615:
	movl	$4, %edi
	jmp	.L1561
.LVL2873:
	.p2align 4,,10
	.p2align 3
.L1614:
	movl	$3, %edi
	jmp	.L1561
.LVL2874:
	.p2align 4,,10
	.p2align 3
.L1613:
	movl	$2, %edi
	jmp	.L1561
.LVL2875:
	.p2align 4,,10
	.p2align 3
.L1612:
	movl	$1, %edi
	jmp	.L1561
.LVL2876:
	.p2align 4,,10
	.p2align 3
.L1694:
	testl	%edx, %edx
	jne	.L1517
.LBE12721:
.LBE12723:
.LBE12744:
.LBB12745:
	.loc 1 72 0
	xorl	%ecx, %ecx
	jmp	.L1518
.LVL2877:
	.p2align 4,,10
	.p2align 3
.L1691:
	testl	%edx, %edx
	jne	.L1502
.LBE12745:
.LBB12746:
	.loc 1 65 0
	xorl	%ecx, %ecx
	jmp	.L1503
.LVL2878:
.L1699:
.LBE12746:
.LBB12747:
.LBB12689:
.LBB12685:
.LBB12682:
	.loc 2 367 0
	leaq	-352(%rbp), %rdi
.LVL2879:
.LEHB53:
	call	_ZN2cv3Mat10deallocateEv
.LVL2880:
.LEHE53:
	jmp	.L1592
.LVL2881:
.L1696:
.LBE12682:
.LBE12685:
.LBE12689:
.LBE12747:
.LBB12748:
.LBB12629:
.LBB12625:
.LBB12622:
	leaq	-448(%rbp), %rdi
.LVL2882:
.LEHB54:
	call	_ZN2cv3Mat10deallocateEv
.LVL2883:
.LEHE54:
	jmp	.L1590
.LVL2884:
.L1700:
.LBE12622:
.LBE12625:
.LBE12629:
.LBE12748:
.LBB12749:
.LBB12690:
	.loc 2 288 0
	movl	-160(%rbp), %eax
.LBB12686:
.LBB12683:
	.loc 2 371 0
	movq	$0, -328(%rbp)
.LVL2885:
.LBE12683:
.LBE12686:
	.loc 2 288 0
	movl	%eax, -352(%rbp)
	jmp	.L1599
.LVL2886:
.L1697:
.LBE12690:
.LBE12749:
.LBB12750:
.LBB12630:
	movl	-256(%rbp), %eax
.LBB12626:
.LBB12623:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL2887:
.LBE12623:
.LBE12626:
	.loc 2 288 0
	movl	%eax, -448(%rbp)
	jmp	.L1598
.LVL2888:
.L1487:
.LBE12630:
.LBE12750:
	.loc 1 47 0 discriminator 3
	movl	$_ZZ12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, %ecx
	movl	$47, %edx
	movl	$.LC45, %esi
	movl	$.LC68, %edi
.LVL2889:
	call	__assert_fail
.LVL2890:
.L1627:
	movq	%rax, %rbx
	jmp	.L1585
.LVL2891:
.L1626:
	movq	%rax, %rbx
	jmp	.L1586
.LVL2892:
.L1695:
	.loc 1 83 0 discriminator 1
	movl	$_ZZ12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, %ecx
	movl	$83, %edx
.LVL2893:
	movl	$.LC45, %esi
	movl	$.LC70, %edi
	call	__assert_fail
.LVL2894:
.L1585:
	.loc 1 87 0 discriminator 2
	leaq	-256(%rbp), %rdi
.LVL2895:
	call	_ZN2cv3MatD1Ev
.LVL2896:
.L1586:
	.loc 1 85 0
	leaq	-352(%rbp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL2897:
	.loc 1 84 0
	leaq	-448(%rbp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL2898:
	movq	%rbx, %rdi
.LEHB55:
	call	_Unwind_Resume
.LVL2899:
.LEHE55:
.L1489:
	.loc 1 54 0 discriminator 3
	movl	$_ZZ12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, %ecx
.LVL2900:
	movl	$54, %edx
.LVL2901:
	movl	$.LC45, %esi
	movl	$.LC69, %edi
.LVL2902:
	call	__assert_fail
.LVL2903:
.L1702:
	.loc 1 116 0 discriminator 1
	movl	$_ZZ12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, %ecx
	movl	$116, %edx
	movl	$.LC45, %esi
	movl	$.LC71, %edi
	call	__assert_fail
.LVL2904:
.L1628:
	movq	%rax, %rbx
	jmp	.L1587
.LVL2905:
.L1703:
	.loc 1 293 0
	call	__stack_chk_fail
.LVL2906:
.L1587:
	.loc 1 88 0 discriminator 2
	leaq	-160(%rbp), %rdi
.LVL2907:
	call	_ZN2cv3MatD1Ev
.LVL2908:
	jmp	.L1586
	.cfi_endproc
.LFE11598:
	.section	.gcc_except_table
.LLSDA11598:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE11598-.LLSDACSB11598
.LLSDACSB11598:
	.uleb128 .LEHB49-.LFB11598
	.uleb128 .LEHE49-.LEHB49
	.uleb128 .L1626-.LFB11598
	.uleb128 0
	.uleb128 .LEHB50-.LFB11598
	.uleb128 .LEHE50-.LEHB50
	.uleb128 .L1627-.LFB11598
	.uleb128 0
	.uleb128 .LEHB51-.LFB11598
	.uleb128 .LEHE51-.LEHB51
	.uleb128 .L1626-.LFB11598
	.uleb128 0
	.uleb128 .LEHB52-.LFB11598
	.uleb128 .LEHE52-.LEHB52
	.uleb128 .L1628-.LFB11598
	.uleb128 0
	.uleb128 .LEHB53-.LFB11598
	.uleb128 .LEHE53-.LEHB53
	.uleb128 .L1628-.LFB11598
	.uleb128 0
	.uleb128 .LEHB54-.LFB11598
	.uleb128 .LEHE54-.LEHB54
	.uleb128 .L1627-.LFB11598
	.uleb128 0
	.uleb128 .LEHB55-.LFB11598
	.uleb128 .LEHE55-.LEHB55
	.uleb128 0
	.uleb128 0
.LLSDACSE11598:
	.section	.text._Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd,comdat
	.size	_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd, .-_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd
	.section	.text.unlikely._Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd,comdat
.LCOLDE74:
	.section	.text._Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd,"axG",@progbits,_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd,comdat
.LHOTE74:
	.section	.text.unlikely
.LCOLDB75:
	.text
.LHOTB75:
	.p2align 4,,15
	.globl	_Z14myGaussianBlurRKN2cv3MatERS0_NS_5Size_IiEEdd
	.type	_Z14myGaussianBlurRKN2cv3MatERS0_NS_5Size_IiEEdd, @function
_Z14myGaussianBlurRKN2cv3MatERS0_NS_5Size_IiEEdd:
.LFB11273:
	.loc 1 296 0
	.cfi_startproc
.LVL2909:
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	.loc 1 296 0
	movq	%fs:40, %rax
	movq	%rax, 8(%rsp)
	xorl	%eax, %eax
.LVL2910:
.LBB12775:
.LBB12776:
	.loc 2 402 0
	movl	(%rdi), %eax
	andl	$4088, %eax
	sarl	$3, %eax
	addl	$1, %eax
.LBE12776:
.LBE12775:
	.loc 1 298 0
	cmpl	$3, %eax
	je	.L1710
	.loc 1 301 0
	cmpl	$2, %eax
	je	.L1711
	.loc 1 304 0
	cmpl	$1, %eax
	je	.L1712
.LVL2911:
.L1704:
	.loc 1 307 0
	movq	8(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L1713
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.LVL2912:
	.p2align 4,,10
	.p2align 3
.L1712:
	.cfi_restore_state
.LBB12777:
.LBB12778:
	.loc 13 1863 0
	movl	(%rdx), %eax
	movl	%eax, (%rsp)
	movl	4(%rdx), %eax
.LBE12778:
.LBE12777:
	.loc 1 305 0
	movq	%rsp, %rdx
.LVL2913:
.LBB12780:
.LBB12779:
	.loc 13 1863 0
	movl	%eax, 4(%rsp)
.LVL2914:
.LBE12779:
.LBE12780:
	.loc 1 305 0
	call	_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd
.LVL2915:
	.loc 1 307 0
	jmp	.L1704
.LVL2916:
	.p2align 4,,10
	.p2align 3
.L1710:
.LBB12781:
.LBB12782:
	.loc 13 1863 0
	movl	(%rdx), %eax
	movl	%eax, (%rsp)
	movl	4(%rdx), %eax
.LBE12782:
.LBE12781:
	.loc 1 299 0
	movq	%rsp, %rdx
.LVL2917:
.LBB12784:
.LBB12783:
	.loc 13 1863 0
	movl	%eax, 4(%rsp)
.LVL2918:
.LBE12783:
.LBE12784:
	.loc 1 299 0
	call	_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd
.LVL2919:
	jmp	.L1704
.LVL2920:
	.p2align 4,,10
	.p2align 3
.L1711:
.LBB12785:
.LBB12786:
.LBB12787:
.LBB12788:
	.loc 13 1863 0
	movl	(%rdx), %eax
	movl	%eax, (%rsp)
	movl	4(%rdx), %eax
.LBE12788:
.LBE12787:
	.loc 1 302 0
	movq	%rsp, %rdx
.LVL2921:
.LBB12790:
.LBB12789:
	.loc 13 1863 0
	movl	%eax, 4(%rsp)
.LVL2922:
.LBE12789:
.LBE12790:
	.loc 1 302 0
	call	_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd
.LVL2923:
	jmp	.L1704
.LVL2924:
.L1713:
.LBE12786:
.LBE12785:
	.loc 1 307 0
	call	__stack_chk_fail
.LVL2925:
	.cfi_endproc
.LFE11273:
	.size	_Z14myGaussianBlurRKN2cv3MatERS0_NS_5Size_IiEEdd, .-_Z14myGaussianBlurRKN2cv3MatERS0_NS_5Size_IiEEdd
	.section	.text.unlikely
.LCOLDE75:
	.text
.LHOTE75:
	.section	.rodata.str1.8
	.align 8
.LC76:
	.string	"src.type() == dst.type() && src.size() == dst.size()"
	.section	.text.unlikely._ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_,comdat
	.align 2
.LCOLDB77:
	.section	.text._ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_,comdat
.LHOTB77:
	.align 2
	.p2align 4,,15
	.weak	_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_
	.type	_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_, @function
_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_:
.LFB11599:
	.loc 1 354 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA11599
.LVL2926:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdx, %rax
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$776, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	.loc 1 354 0
	movq	%rdi, -800(%rbp)
	movq	%rdx, -808(%rbp)
	movq	%fs:40, %rbx
	movq	%rbx, -56(%rbp)
	xorl	%ebx, %ebx
	.loc 1 356 0
	movl	8(%rsi), %ebx
	.loc 1 359 0
	movl	(%rsi), %edx
.LVL2927:
	.loc 1 356 0
	movl	%ebx, -788(%rbp)
.LVL2928:
	.loc 1 357 0
	movl	12(%rsi), %ebx
.LVL2929:
	.loc 1 359 0
	andl	$4095, %edx
	.loc 1 358 0
	leal	8(%rbx), %edi
.LVL2930:
	movl	%edi, -792(%rbp)
.LVL2931:
	.loc 1 359 0
	movq	%rax, %rdi
.LVL2932:
	movl	(%rax), %eax
.LVL2933:
	movl	%eax, -816(%rbp)
	andl	$4095, %eax
	cmpl	%eax, %edx
	jne	.L1715
.LBB12913:
.LBB12914:
	.loc 2 713 0 discriminator 2
	movq	64(%rdi), %rax
.LBE12914:
.LBE12913:
.LBB12915:
.LBB12916:
	movq	64(%rsi), %rdx
	movq	%rsi, %r14
.LVL2934:
.LBE12916:
.LBE12915:
.LBB12917:
.LBB12918:
	.loc 13 1893 0 discriminator 2
	movl	(%rax), %edi
.LVL2935:
	cmpl	%edi, (%rdx)
	jne	.L1715
	movl	4(%rax), %eax
	cmpl	%eax, 4(%rdx)
	je	.L1716
.LVL2936:
.L1715:
.LBE12918:
.LBE12917:
	.loc 1 359 0 discriminator 7
	movl	$_ZZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_E19__PRETTY_FUNCTION__, %ecx
	movl	$359, %edx
	movl	$.LC45, %esi
.LVL2937:
	movl	$.LC76, %edi
	call	__assert_fail
.LVL2938:
	.p2align 4,,10
	.p2align 3
.L1716:
.LBB12919:
.LBB12920:
.LBB12921:
	.loc 2 709 0
	leaq	-640(%rbp), %r12
.LVL2939:
.LBE12921:
.LBE12920:
.LBE12919:
.LBB12942:
.LBB12943:
.LBB12944:
	leaq	-544(%rbp), %r13
.LBE12944:
.LBE12943:
.LBE12942:
.LBB12963:
.LBB12964:
.LBB12965:
	leaq	-448(%rbp), %r15
.LBE12965:
.LBE12964:
.LBE12963:
.LBB12984:
.LBB12924:
.LBB12925:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE12925:
.LBE12924:
.LBB12928:
.LBB12929:
	.loc 2 62 0
	movq	$0, -592(%rbp)
.LBE12929:
.LBE12928:
.LBB12934:
.LBB12922:
	.loc 2 709 0
	leaq	8(%r12), %rax
.LBE12922:
.LBE12934:
.LBB12935:
.LBB12930:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm5
	.loc 2 62 0
	movq	$0, -600(%rbp)
	movq	$0, -608(%rbp)
.LBE12930:
.LBE12935:
.LBE12984:
.LBB12985:
.LBB12986:
.LBB12987:
	.loc 2 353 0
	leaq	-688(%rbp), %rdx
.LBE12987:
.LBE12986:
.LBE12985:
.LBB13006:
.LBB12936:
.LBB12923:
	.loc 2 709 0
	movq	%rax, -576(%rbp)
.LVL2940:
.LBE12923:
.LBE12936:
.LBB12937:
.LBB12926:
	.loc 2 738 0
	leaq	80(%r12), %rax
.LBE12926:
.LBE12937:
.LBB12938:
.LBB12931:
	.loc 2 62 0
	movq	$0, -624(%rbp)
.LBE12931:
.LBE12938:
.LBB12939:
.LBB12927:
	.loc 2 738 0
	movaps	%xmm0, -560(%rbp)
.LVL2941:
	movq	%rax, -568(%rbp)
.LBE12927:
.LBE12939:
.LBE13006:
.LBB13007:
.LBB12947:
.LBB12945:
	.loc 2 709 0
	leaq	8(%r13), %rax
.LBE12945:
.LBE12947:
.LBE13007:
.LBB13008:
.LBB12940:
.LBB12932:
	.loc 2 63 0
	movq	$0, -616(%rbp)
	.loc 2 60 0
	movaps	%xmm5, -640(%rbp)
.LBE12932:
.LBE12940:
.LBE13008:
.LBB13009:
.LBB12991:
.LBB12988:
	.loc 2 353 0
	leaq	-352(%rbp), %rdi
	movl	$5, %ecx
	movl	$2, %esi
.LVL2942:
.LBE12988:
.LBE12991:
.LBE13009:
.LBB13010:
.LBB12948:
.LBB12949:
	.loc 2 738 0
	movaps	%xmm0, -464(%rbp)
.LBE12949:
.LBE12948:
.LBB12952:
.LBB12953:
	.loc 2 60 0
	movaps	%xmm5, -544(%rbp)
.LBE12953:
.LBE12952:
.LBE13010:
.LBB13011:
.LBB12968:
.LBB12969:
	.loc 2 738 0
	movaps	%xmm0, -368(%rbp)
.LBE12969:
.LBE12968:
.LBB12972:
.LBB12973:
	.loc 2 60 0
	movaps	%xmm5, -448(%rbp)
.LBE12973:
.LBE12972:
.LBE13011:
.LBB13012:
.LBB12957:
.LBB12946:
	.loc 2 709 0
	movq	%rax, -480(%rbp)
.LBE12946:
.LBE12957:
.LBB12958:
.LBB12950:
	.loc 2 738 0
	leaq	80(%r13), %rax
.LBE12950:
.LBE12958:
.LBE13012:
.LBB13013:
.LBB12941:
.LBB12933:
	.loc 2 64 0
	movq	$0, -584(%rbp)
.LVL2943:
.LBE12933:
.LBE12941:
.LBE13013:
.LBB13014:
.LBB12959:
.LBB12954:
	.loc 2 62 0
	movq	$0, -496(%rbp)
	movq	$0, -504(%rbp)
.LBE12954:
.LBE12959:
.LBB12960:
.LBB12951:
	.loc 2 738 0
	movq	%rax, -472(%rbp)
.LBE12951:
.LBE12960:
.LBE13014:
.LBB13015:
.LBB12977:
.LBB12966:
	.loc 2 709 0
	leaq	8(%r15), %rax
.LBE12966:
.LBE12977:
.LBE13015:
.LBB13016:
.LBB12961:
.LBB12955:
	.loc 2 62 0
	movq	$0, -512(%rbp)
	movq	$0, -528(%rbp)
	.loc 2 63 0
	movq	$0, -520(%rbp)
.LBE12955:
.LBE12961:
.LBE13016:
.LBB13017:
.LBB12978:
.LBB12967:
	.loc 2 709 0
	movq	%rax, -384(%rbp)
.LBE12967:
.LBE12978:
.LBB12979:
.LBB12970:
	.loc 2 738 0
	leaq	80(%r15), %rax
.LBE12970:
.LBE12979:
.LBE13017:
.LBB13018:
.LBB12962:
.LBB12956:
	.loc 2 64 0
	movq	$0, -488(%rbp)
.LVL2944:
.LBE12956:
.LBE12962:
.LBE13018:
.LBB13019:
.LBB12980:
.LBB12974:
	.loc 2 62 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
.LBE12974:
.LBE12980:
.LBB12981:
.LBB12971:
	.loc 2 738 0
	movq	%rax, -376(%rbp)
.LBE12971:
.LBE12981:
.LBE13019:
.LBB13020:
.LBB12992:
.LBB12993:
	.loc 2 709 0
	leaq	-352(%rbp), %rax
.LBE12993:
.LBE12992:
.LBE13020:
.LBB13021:
.LBB12982:
.LBB12975:
	.loc 2 62 0
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
	.loc 2 63 0
	movq	$0, -424(%rbp)
.LBE12975:
.LBE12982:
.LBE13021:
.LBB13022:
.LBB12996:
.LBB12994:
	.loc 2 709 0
	addq	$8, %rax
.LBE12994:
.LBE12996:
.LBE13022:
.LBB13023:
.LBB12983:
.LBB12976:
	.loc 2 64 0
	movq	$0, -392(%rbp)
.LVL2945:
.LBE12976:
.LBE12983:
.LBE13023:
.LBB13024:
.LBB12997:
.LBB12995:
	.loc 2 709 0
	movq	%rax, -288(%rbp)
.LVL2946:
.LBE12995:
.LBE12997:
.LBB12998:
.LBB12999:
	.loc 2 738 0
	leaq	-352(%rbp), %rax
	addq	$80, %rax
	movq	%rax, -280(%rbp)
.LBE12999:
.LBE12998:
.LBB13001:
.LBB12989:
	.loc 2 352 0
	movl	-788(%rbp), %eax
.LBE12989:
.LBE13001:
.LBB13002:
.LBB13000:
	.loc 2 738 0
	movaps	%xmm0, -272(%rbp)
.LVL2947:
.LBE13000:
.LBE13002:
.LBB13003:
.LBB13004:
	.loc 2 62 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
	.loc 2 60 0
	movaps	%xmm5, -352(%rbp)
	.loc 2 62 0
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
	.loc 2 63 0
	movq	$0, -328(%rbp)
	.loc 2 64 0
	movq	$0, -296(%rbp)
.LVL2948:
.LBE13004:
.LBE13003:
.LBB13005:
.LBB12990:
	.loc 2 352 0
	movl	%eax, -688(%rbp)
	movl	%ebx, -684(%rbp)
.LEHB56:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL2949:
.LEHE56:
.LBE12990:
.LBE13005:
.LBE13024:
.LBB13025:
.LBB13026:
	.loc 2 285 0
	movq	-328(%rbp), %rax
	testq	%rax, %rax
	je	.L1718
	.loc 2 286 0
	lock addl	$1, (%rax)
.L1718:
.LVL2950:
.LBB13027:
.LBB13028:
	.loc 2 366 0
	movq	-616(%rbp), %rax
	testq	%rax, %rax
	je	.L1795
	lock subl	$1, (%rax)
	je	.L1882
.L1795:
.LBB13029:
	.loc 2 369 0
	movl	-636(%rbp), %ecx
.LBE13029:
	.loc 2 368 0
	movq	$0, -592(%rbp)
	movq	$0, -600(%rbp)
	movq	$0, -608(%rbp)
	movq	$0, -624(%rbp)
.LVL2951:
.LBB13030:
	.loc 2 369 0
	testl	%ecx, %ecx
	jle	.L1883
	movq	-576(%rbp), %rdx
	xorl	%eax, %eax
.LVL2952:
	.p2align 4,,10
	.p2align 3
.L1721:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-636(%rbp), %ecx
	addl	$1, %eax
.LVL2953:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L1721
.LBE13030:
.LBE13028:
.LBE13027:
	.loc 2 288 0
	movl	-352(%rbp), %eax
.LVL2954:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB13034:
.LBB13031:
	.loc 2 371 0
	movq	$0, -616(%rbp)
.LVL2955:
.LBE13031:
.LBE13034:
	.loc 2 288 0
	movl	%eax, -640(%rbp)
	.loc 2 289 0
	jg	.L1722
.L1810:
	movl	-348(%rbp), %eax
	cmpl	$2, %eax
	jle	.L1884
.L1722:
	.loc 2 298 0
	leaq	-352(%rbp), %rsi
.LVL2956:
	movq	%r12, %rdi
.LEHB57:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL2957:
.LEHE57:
.L1723:
	.loc 2 299 0
	movdqa	-336(%rbp), %xmm0
	.loc 2 303 0
	movq	-328(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -624(%rbp)
.LBE13026:
.LBE13025:
.LBB13041:
.LBB13042:
.LBB13043:
.LBB13044:
	.loc 2 366 0
	testq	%rax, %rax
.LBE13044:
.LBE13043:
.LBE13042:
.LBE13041:
.LBB13052:
.LBB13037:
	.loc 2 299 0
	movdqa	-320(%rbp), %xmm0
	movaps	%xmm0, -608(%rbp)
	movdqa	-304(%rbp), %xmm0
	movaps	%xmm0, -592(%rbp)
.LVL2958:
.LBE13037:
.LBE13052:
.LBB13053:
.LBB13051:
.LBB13049:
.LBB13047:
	.loc 2 366 0
	je	.L1798
	lock subl	$1, (%rax)
	jne	.L1798
	.loc 2 367 0
	leaq	-352(%rbp), %rdi
.LVL2959:
	call	_ZN2cv3Mat10deallocateEv
.LVL2960:
.L1798:
.LBB13045:
	.loc 2 369 0
	movl	-348(%rbp), %r11d
.LBE13045:
	.loc 2 368 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
.LVL2961:
.LBB13046:
	.loc 2 369 0
	testl	%r11d, %r11d
	jle	.L1729
	movq	-288(%rbp), %rdx
	xorl	%eax, %eax
.LVL2962:
	.p2align 4,,10
	.p2align 3
.L1730:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL2963:
	addq	$4, %rdx
	cmpl	%eax, -348(%rbp)
	jg	.L1730
.LVL2964:
.L1729:
.LBE13046:
.LBE13047:
.LBE13049:
	.loc 2 277 0
	movq	-280(%rbp), %rdi
	leaq	-352(%rbp), %rax
.LVL2965:
.LBB13050:
.LBB13048:
	.loc 2 371 0
	movq	$0, -328(%rbp)
.LVL2966:
.LBE13048:
.LBE13050:
	.loc 2 277 0
	addq	$80, %rax
.LVL2967:
	cmpq	%rax, %rdi
	je	.L1728
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL2968:
.L1728:
.LBE13051:
.LBE13053:
.LBB13054:
.LBB13055:
.LBB13056:
	.loc 2 709 0
	leaq	-256(%rbp), %rax
.LVL2969:
.LBE13056:
.LBE13055:
.LBB13059:
.LBB13060:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE13060:
.LBE13059:
.LBB13064:
.LBB13065:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm3
.LBE13065:
.LBE13064:
.LBB13068:
.LBB13069:
	.loc 2 353 0
	leaq	-672(%rbp), %rdx
.LBE13069:
.LBE13068:
.LBB13074:
.LBB13057:
	.loc 2 709 0
	addq	$8, %rax
.LVL2970:
.LBE13057:
.LBE13074:
.LBB13075:
.LBB13070:
	.loc 2 353 0
	leaq	-256(%rbp), %rdi
.LVL2971:
	movl	$5, %ecx
.LBE13070:
.LBE13075:
.LBB13076:
.LBB13058:
	.loc 2 709 0
	movq	%rax, -192(%rbp)
.LVL2972:
.LBE13058:
.LBE13076:
.LBB13077:
.LBB13061:
	.loc 2 738 0
	leaq	-256(%rbp), %rax
.LBE13061:
.LBE13077:
.LBB13078:
.LBB13071:
	.loc 2 353 0
	movl	$2, %esi
.LBE13071:
.LBE13078:
.LBB13079:
.LBB13062:
	.loc 2 738 0
	movaps	%xmm0, -176(%rbp)
.LVL2973:
	addq	$80, %rax
.LBE13062:
.LBE13079:
.LBB13080:
.LBB13066:
	.loc 2 62 0
	movq	$0, -208(%rbp)
	movq	$0, -216(%rbp)
	.loc 2 60 0
	movaps	%xmm3, -256(%rbp)
.LBE13066:
.LBE13080:
.LBB13081:
.LBB13063:
	.loc 2 738 0
	movq	%rax, -184(%rbp)
.LBE13063:
.LBE13081:
.LBB13082:
.LBB13072:
	.loc 2 352 0
	movl	-788(%rbp), %eax
.LBE13072:
.LBE13082:
.LBB13083:
.LBB13067:
	.loc 2 62 0
	movq	$0, -224(%rbp)
	movq	$0, -240(%rbp)
	.loc 2 63 0
	movq	$0, -232(%rbp)
	.loc 2 64 0
	movq	$0, -200(%rbp)
.LVL2974:
.LBE13067:
.LBE13083:
.LBB13084:
.LBB13073:
	.loc 2 352 0
	movl	%eax, -672(%rbp)
	movl	%ebx, -668(%rbp)
.LEHB58:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL2975:
.LEHE58:
.LBE13073:
.LBE13084:
.LBE13054:
.LBB13085:
.LBB13086:
	.loc 2 285 0
	movq	-232(%rbp), %rax
	testq	%rax, %rax
	je	.L1731
	.loc 2 286 0
	lock addl	$1, (%rax)
.L1731:
.LVL2976:
.LBB13087:
.LBB13088:
	.loc 2 366 0
	movq	-520(%rbp), %rax
	testq	%rax, %rax
	je	.L1799
	lock subl	$1, (%rax)
	je	.L1885
.L1799:
.LBB13089:
	.loc 2 369 0
	movl	-540(%rbp), %edx
.LBE13089:
	.loc 2 368 0
	movq	$0, -496(%rbp)
	movq	$0, -504(%rbp)
	movq	$0, -512(%rbp)
	movq	$0, -528(%rbp)
.LVL2977:
.LBB13090:
	.loc 2 369 0
	testl	%edx, %edx
	jle	.L1886
	movq	-480(%rbp), %rdx
	xorl	%eax, %eax
.LVL2978:
	.p2align 4,,10
	.p2align 3
.L1734:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-540(%rbp), %ecx
	addl	$1, %eax
.LVL2979:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L1734
.LBE13090:
.LBE13088:
.LBE13087:
	.loc 2 288 0
	movl	-256(%rbp), %eax
.LVL2980:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB13094:
.LBB13091:
	.loc 2 371 0
	movq	$0, -520(%rbp)
.LVL2981:
.LBE13091:
.LBE13094:
	.loc 2 288 0
	movl	%eax, -544(%rbp)
	.loc 2 289 0
	jg	.L1735
.L1811:
	movl	-252(%rbp), %eax
	cmpl	$2, %eax
	jle	.L1887
.L1735:
	.loc 2 298 0
	leaq	-256(%rbp), %rsi
.LVL2982:
	movq	%r13, %rdi
.LEHB59:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL2983:
.LEHE59:
.L1736:
	.loc 2 299 0
	movdqa	-240(%rbp), %xmm0
	.loc 2 303 0
	movq	-232(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -528(%rbp)
.LBE13086:
.LBE13085:
.LBB13101:
.LBB13102:
.LBB13103:
.LBB13104:
	.loc 2 366 0
	testq	%rax, %rax
.LBE13104:
.LBE13103:
.LBE13102:
.LBE13101:
.LBB13112:
.LBB13097:
	.loc 2 299 0
	movdqa	-224(%rbp), %xmm0
	movaps	%xmm0, -512(%rbp)
	movdqa	-208(%rbp), %xmm0
	movaps	%xmm0, -496(%rbp)
.LVL2984:
.LBE13097:
.LBE13112:
.LBB13113:
.LBB13111:
.LBB13109:
.LBB13107:
	.loc 2 366 0
	je	.L1802
	lock subl	$1, (%rax)
	jne	.L1802
	.loc 2 367 0
	leaq	-256(%rbp), %rdi
.LVL2985:
	call	_ZN2cv3Mat10deallocateEv
.LVL2986:
.L1802:
.LBB13105:
	.loc 2 369 0
	movl	-252(%rbp), %r10d
.LBE13105:
	.loc 2 368 0
	movq	$0, -208(%rbp)
	movq	$0, -216(%rbp)
	movq	$0, -224(%rbp)
	movq	$0, -240(%rbp)
.LVL2987:
.LBB13106:
	.loc 2 369 0
	testl	%r10d, %r10d
	jle	.L1742
	movq	-192(%rbp), %rdx
	xorl	%eax, %eax
.LVL2988:
	.p2align 4,,10
	.p2align 3
.L1743:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL2989:
	addq	$4, %rdx
	cmpl	%eax, -252(%rbp)
	jg	.L1743
.LVL2990:
.L1742:
.LBE13106:
.LBE13107:
.LBE13109:
	.loc 2 277 0
	movq	-184(%rbp), %rdi
	leaq	-256(%rbp), %rax
.LVL2991:
.LBB13110:
.LBB13108:
	.loc 2 371 0
	movq	$0, -232(%rbp)
.LVL2992:
.LBE13108:
.LBE13110:
	.loc 2 277 0
	addq	$80, %rax
.LVL2993:
	cmpq	%rax, %rdi
	je	.L1741
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL2994:
.L1741:
.LBE13111:
.LBE13113:
.LBB13114:
.LBB13115:
.LBB13116:
	.loc 2 709 0
	leaq	-160(%rbp), %rax
.LVL2995:
.LBE13116:
.LBE13115:
.LBB13119:
.LBB13120:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE13120:
.LBE13119:
.LBB13124:
.LBB13125:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm4
.LBE13125:
.LBE13124:
.LBB13128:
.LBB13129:
	.loc 2 353 0
	leaq	-656(%rbp), %rdx
.LBE13129:
.LBE13128:
.LBB13134:
.LBB13117:
	.loc 2 709 0
	addq	$8, %rax
.LVL2996:
.LBE13117:
.LBE13134:
.LBB13135:
.LBB13130:
	.loc 2 353 0
	leaq	-160(%rbp), %rdi
.LVL2997:
	movl	$5, %ecx
.LBE13130:
.LBE13135:
.LBB13136:
.LBB13118:
	.loc 2 709 0
	movq	%rax, -96(%rbp)
.LVL2998:
.LBE13118:
.LBE13136:
.LBB13137:
.LBB13121:
	.loc 2 738 0
	leaq	-160(%rbp), %rax
.LBE13121:
.LBE13137:
.LBB13138:
.LBB13131:
	.loc 2 353 0
	movl	$2, %esi
.LBE13131:
.LBE13138:
.LBB13139:
.LBB13122:
	.loc 2 738 0
	movaps	%xmm0, -80(%rbp)
.LVL2999:
	addq	$80, %rax
.LBE13122:
.LBE13139:
.LBB13140:
.LBB13126:
	.loc 2 62 0
	movq	$0, -112(%rbp)
	movq	$0, -120(%rbp)
	.loc 2 60 0
	movaps	%xmm4, -160(%rbp)
.LBE13126:
.LBE13140:
.LBB13141:
.LBB13123:
	.loc 2 738 0
	movq	%rax, -88(%rbp)
.LBE13123:
.LBE13141:
.LBB13142:
.LBB13132:
	.loc 2 352 0
	movl	-792(%rbp), %eax
.LBE13132:
.LBE13142:
.LBB13143:
.LBB13127:
	.loc 2 62 0
	movq	$0, -128(%rbp)
	movq	$0, -144(%rbp)
	.loc 2 63 0
	movq	$0, -136(%rbp)
	.loc 2 64 0
	movq	$0, -104(%rbp)
.LVL3000:
.LBE13127:
.LBE13143:
.LBB13144:
.LBB13133:
	.loc 2 352 0
	movl	$4, -656(%rbp)
	movl	%eax, -652(%rbp)
.LEHB60:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL3001:
.LEHE60:
.LBE13133:
.LBE13144:
.LBE13114:
.LBB13145:
.LBB13146:
	.loc 2 285 0
	movq	-136(%rbp), %rax
	testq	%rax, %rax
	je	.L1744
	.loc 2 286 0
	lock addl	$1, (%rax)
.L1744:
.LVL3002:
.LBB13147:
.LBB13148:
	.loc 2 366 0
	movq	-424(%rbp), %rax
	testq	%rax, %rax
	je	.L1803
	lock subl	$1, (%rax)
	je	.L1888
.L1803:
.LBB13149:
	.loc 2 369 0
	movl	-444(%rbp), %eax
.LBE13149:
	.loc 2 368 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
.LVL3003:
.LBB13150:
	.loc 2 369 0
	testl	%eax, %eax
	jle	.L1889
	movq	-384(%rbp), %rdx
	xorl	%eax, %eax
.LVL3004:
	.p2align 4,,10
	.p2align 3
.L1747:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-444(%rbp), %ecx
	addl	$1, %eax
.LVL3005:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L1747
.LBE13150:
.LBE13148:
.LBE13147:
	.loc 2 288 0
	movl	-160(%rbp), %eax
.LVL3006:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB13154:
.LBB13151:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL3007:
.LBE13151:
.LBE13154:
	.loc 2 288 0
	movl	%eax, -448(%rbp)
	.loc 2 289 0
	jg	.L1748
.L1812:
	movl	-156(%rbp), %eax
	cmpl	$2, %eax
	jle	.L1890
.L1748:
	.loc 2 298 0
	leaq	-160(%rbp), %rsi
.LVL3008:
	movq	%r15, %rdi
.LEHB61:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL3009:
.LEHE61:
.L1749:
	.loc 2 299 0
	movdqa	-144(%rbp), %xmm0
	.loc 2 303 0
	movq	-136(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -432(%rbp)
.LBE13146:
.LBE13145:
.LBB13161:
.LBB13162:
.LBB13163:
.LBB13164:
	.loc 2 366 0
	testq	%rax, %rax
.LBE13164:
.LBE13163:
.LBE13162:
.LBE13161:
.LBB13172:
.LBB13157:
	.loc 2 299 0
	movdqa	-128(%rbp), %xmm0
	movaps	%xmm0, -416(%rbp)
	movdqa	-112(%rbp), %xmm0
	movaps	%xmm0, -400(%rbp)
.LVL3010:
.LBE13157:
.LBE13172:
.LBB13173:
.LBB13171:
.LBB13169:
.LBB13167:
	.loc 2 366 0
	je	.L1806
	lock subl	$1, (%rax)
	jne	.L1806
	.loc 2 367 0
	leaq	-160(%rbp), %rdi
.LVL3011:
	call	_ZN2cv3Mat10deallocateEv
.LVL3012:
.L1806:
.LBB13165:
	.loc 2 369 0
	movl	-156(%rbp), %r9d
.LBE13165:
	.loc 2 368 0
	movq	$0, -112(%rbp)
	movq	$0, -120(%rbp)
	movq	$0, -128(%rbp)
	movq	$0, -144(%rbp)
.LVL3013:
.LBB13166:
	.loc 2 369 0
	testl	%r9d, %r9d
	jle	.L1755
	movq	-96(%rbp), %rdx
	xorl	%eax, %eax
.LVL3014:
	.p2align 4,,10
	.p2align 3
.L1756:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3015:
	addq	$4, %rdx
	cmpl	%eax, -156(%rbp)
	jg	.L1756
.LVL3016:
.L1755:
.LBE13166:
.LBE13167:
.LBE13169:
	.loc 2 277 0
	movq	-88(%rbp), %rdi
	leaq	-160(%rbp), %rax
.LVL3017:
.LBB13170:
.LBB13168:
	.loc 2 371 0
	movq	$0, -136(%rbp)
.LVL3018:
.LBE13168:
.LBE13170:
	.loc 2 277 0
	addq	$80, %rax
.LVL3019:
	cmpq	%rax, %rdi
	je	.L1754
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3020:
.L1754:
.LBE13171:
.LBE13173:
	.loc 1 383 0
	movl	%ebx, %eax
	.loc 1 384 0
	movslq	%ebx, %rcx
	movq	16(%r14), %rdx
	.loc 1 383 0
	sarl	$6, %eax
	.loc 1 384 0
	leaq	0(,%rcx,4), %rsi
	.loc 1 383 0
	sall	$4, %eax
	movl	%eax, -792(%rbp)
.LVL3021:
	.loc 1 384 0
	leaq	18(%rsi), %rax
.LVL3022:
	andq	$-16, %rax
	subq	%rax, %rsp
	leaq	3(%rsp), %rdi
	shrq	$2, %rdi
.LBB13174:
	.loc 1 386 0
	testl	%ebx, %ebx
.LBE13174:
	.loc 1 384 0
	leaq	0(,%rdi,4), %rax
.LVL3023:
.LBB13175:
	.loc 1 386 0
	jle	.L1767
	addq	%rax, %rsi
	cmpq	%rsi, %rdx
	setnb	%sil
	addq	%rdx, %rcx
	cmpq	%rcx, %rax
	setnb	%cl
	orb	%cl, %sil
	je	.L1760
	cmpl	$20, %ebx
	jbe	.L1760
	movq	%rax, %rsi
	andl	$15, %esi
	shrq	$2, %rsi
	negq	%rsi
	andl	$3, %esi
	cmpl	%ebx, %esi
	cmova	%ebx, %esi
	xorl	%ecx, %ecx
	testl	%esi, %esi
	je	.L1761
	.loc 1 387 0
	movzbl	(%rdx), %ecx
	pxor	%xmm0, %xmm0
	cmpl	$1, %esi
	cvtsi2ss	%ecx, %xmm0
	.loc 1 386 0
	movl	$1, %ecx
	.loc 1 387 0
	movss	%xmm0, 0(,%rdi,4)
.LVL3024:
	je	.L1761
	movzbl	1(%rdx), %ecx
	pxor	%xmm0, %xmm0
	cmpl	$2, %esi
	cvtsi2ss	%ecx, %xmm0
	.loc 1 386 0
	movl	$2, %ecx
	.loc 1 387 0
	movss	%xmm0, 4(,%rdi,4)
.LVL3025:
	je	.L1761
	movzbl	2(%rdx), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2ss	%ecx, %xmm0
	.loc 1 386 0
	movl	$3, %ecx
	.loc 1 387 0
	movss	%xmm0, 8(,%rdi,4)
.LVL3026:
.L1761:
	movl	%ebx, %r8d
	movl	%esi, %edi
	.loc 1 386 0
	xorl	%r11d, %r11d
	subl	%esi, %r8d
	leaq	(%rax,%rdi,4), %r10
	addq	%rdx, %rdi
	leal	-16(%r8), %esi
	shrl	$4, %esi
	addl	$1, %esi
	movl	%esi, %r9d
	sall	$4, %r9d
.L1764:
	.loc 1 387 0 discriminator 2
	movdqu	(%rdi), %xmm0
	addl	$1, %r11d
	addq	$64, %r10
	addq	$16, %rdi
	pmovzxbw	%xmm0, %xmm1
	psrldq	$8, %xmm0
	pmovzxbw	%xmm0, %xmm0
	pmovsxwd	%xmm1, %xmm2
	psrldq	$8, %xmm1
	pmovsxwd	%xmm1, %xmm1
	cvtdq2ps	%xmm2, %xmm2
	movaps	%xmm2, -64(%r10)
	cvtdq2ps	%xmm1, %xmm1
	movaps	%xmm1, -48(%r10)
	pmovsxwd	%xmm0, %xmm1
	psrldq	$8, %xmm0
	pmovsxwd	%xmm0, %xmm0
	cvtdq2ps	%xmm1, %xmm1
	movaps	%xmm1, -32(%r10)
	cvtdq2ps	%xmm0, %xmm0
	movaps	%xmm0, -16(%r10)
	cmpl	%r11d, %esi
	ja	.L1764
	addl	%r9d, %ecx
	cmpl	%r8d, %r9d
	je	.L1767
.LVL3027:
	.loc 1 387 0 is_stmt 0
	movslq	%ecx, %rdi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rdi), %esi
	cvtsi2ss	%esi, %xmm0
	.loc 1 386 0 is_stmt 1
	leal	1(%rcx), %esi
.LVL3028:
	cmpl	%esi, %ebx
	.loc 1 387 0
	movss	%xmm0, (%rax,%rdi,4)
	.loc 1 386 0
	jle	.L1767
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	2(%rcx), %esi
.LVL3029:
	cmpl	%esi, %ebx
	jle	.L1767
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	3(%rcx), %esi
.LVL3030:
	cmpl	%esi, %ebx
	jle	.L1767
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	4(%rcx), %esi
.LVL3031:
	cmpl	%esi, %ebx
	jle	.L1767
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	5(%rcx), %esi
.LVL3032:
	cmpl	%esi, %ebx
	jle	.L1767
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	6(%rcx), %esi
.LVL3033:
	cmpl	%esi, %ebx
	jle	.L1767
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	7(%rcx), %esi
.LVL3034:
	cmpl	%esi, %ebx
	jle	.L1767
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	8(%rcx), %esi
.LVL3035:
	cmpl	%esi, %ebx
	jle	.L1767
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	9(%rcx), %esi
.LVL3036:
	cmpl	%esi, %ebx
	jle	.L1767
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	10(%rcx), %esi
.LVL3037:
	cmpl	%esi, %ebx
	jle	.L1767
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	11(%rcx), %esi
.LVL3038:
	cmpl	%esi, %ebx
	jle	.L1767
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	12(%rcx), %esi
.LVL3039:
	cmpl	%esi, %ebx
	jle	.L1767
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	13(%rcx), %esi
.LVL3040:
	cmpl	%esi, %ebx
	jle	.L1767
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	.loc 1 386 0
	addl	$14, %ecx
	cmpl	%ecx, %ebx
	.loc 1 387 0
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
.LVL3041:
	.loc 1 386 0
	jle	.L1767
	.loc 1 387 0
	movslq	%ecx, %rcx
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rcx), %edx
	cvtsi2ss	%edx, %xmm0
	movss	%xmm0, (%rax,%rcx,4)
.L1767:
	movq	%rax, -816(%rbp)
.LBE13175:
	.loc 1 390 0
	call	omp_get_wtime
.LVL3042:
.LBB13176:
	.loc 1 392 0
	movq	-816(%rbp), %rax
	pxor	%xmm0, %xmm0
	leaq	-784(%rbp), %rsi
	xorl	%ecx, %ecx
	movl	$4, %edx
	movl	$_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_._omp_fn.3, %edi
	movq	%r14, -752(%rbp)
	movq	%r12, -736(%rbp)
	movq	%rax, -760(%rbp)
	movl	-792(%rbp), %eax
	movaps	%xmm0, -784(%rbp)
	movq	%r15, -728(%rbp)
	movq	%r13, -720(%rbp)
	movl	%eax, -704(%rbp)
	movq	-800(%rbp), %rax
	movl	%ebx, -708(%rbp)
	movq	%rax, -768(%rbp)
	movq	-808(%rbp), %rax
	movq	%rax, -744(%rbp)
	movl	-788(%rbp), %eax
	movl	%eax, -712(%rbp)
	call	GOMP_parallel
.LVL3043:
.LBE13176:
	.loc 1 608 0
	call	omp_get_wtime
.LVL3044:
.LBB13177:
.LBB13178:
.LBB13179:
.LBB13180:
	.loc 2 366 0
	movq	-424(%rbp), %rax
	testq	%rax, %rax
	je	.L1807
	lock subl	$1, (%rax)
	jne	.L1807
	.loc 2 367 0
	movq	%r15, %rdi
	call	_ZN2cv3Mat10deallocateEv
.LVL3045:
.L1807:
.LBB13181:
	.loc 2 369 0
	movl	-444(%rbp), %r8d
.LBE13181:
	.loc 2 368 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
.LVL3046:
.LBB13182:
	.loc 2 369 0
	testl	%r8d, %r8d
	jle	.L1774
	movq	-384(%rbp), %rdx
	xorl	%eax, %eax
.LVL3047:
	.p2align 4,,10
	.p2align 3
.L1775:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3048:
	addq	$4, %rdx
	cmpl	%eax, -444(%rbp)
	jg	.L1775
.LVL3049:
.L1774:
.LBE13182:
.LBE13180:
.LBE13179:
	.loc 2 277 0
	movq	-376(%rbp), %rdi
	leaq	80(%r15), %rax
.LBB13184:
.LBB13183:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL3050:
.LBE13183:
.LBE13184:
	.loc 2 277 0
	cmpq	%rax, %rdi
	je	.L1773
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3051:
.L1773:
.LBE13178:
.LBE13177:
.LBB13185:
.LBB13186:
.LBB13187:
.LBB13188:
	.loc 2 366 0
	movq	-520(%rbp), %rax
	testq	%rax, %rax
	je	.L1808
	lock subl	$1, (%rax)
	jne	.L1808
	.loc 2 367 0
	movq	%r13, %rdi
	call	_ZN2cv3Mat10deallocateEv
.LVL3052:
.L1808:
.LBB13189:
	.loc 2 369 0
	movl	-540(%rbp), %edi
.LBE13189:
	.loc 2 368 0
	movq	$0, -496(%rbp)
	movq	$0, -504(%rbp)
	movq	$0, -512(%rbp)
	movq	$0, -528(%rbp)
.LVL3053:
.LBB13190:
	.loc 2 369 0
	testl	%edi, %edi
	jle	.L1781
	movq	-480(%rbp), %rdx
	xorl	%eax, %eax
.LVL3054:
	.p2align 4,,10
	.p2align 3
.L1782:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3055:
	addq	$4, %rdx
	cmpl	%eax, -540(%rbp)
	jg	.L1782
.LVL3056:
.L1781:
.LBE13190:
.LBE13188:
.LBE13187:
	.loc 2 277 0
	movq	-472(%rbp), %rdi
	addq	$80, %r13
.LVL3057:
.LBB13192:
.LBB13191:
	.loc 2 371 0
	movq	$0, -520(%rbp)
.LVL3058:
.LBE13191:
.LBE13192:
	.loc 2 277 0
	cmpq	%r13, %rdi
	je	.L1780
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3059:
.L1780:
.LBE13186:
.LBE13185:
.LBB13193:
.LBB13194:
.LBB13195:
.LBB13196:
	.loc 2 366 0
	movq	-616(%rbp), %rax
	testq	%rax, %rax
	je	.L1809
	lock subl	$1, (%rax)
	jne	.L1809
	.loc 2 367 0
	movq	%r12, %rdi
	call	_ZN2cv3Mat10deallocateEv
.LVL3060:
.L1809:
.LBB13197:
	.loc 2 369 0
	movl	-636(%rbp), %esi
.LBE13197:
	.loc 2 368 0
	movq	$0, -592(%rbp)
	movq	$0, -600(%rbp)
	movq	$0, -608(%rbp)
	movq	$0, -624(%rbp)
.LVL3061:
.LBB13198:
	.loc 2 369 0
	testl	%esi, %esi
	jle	.L1788
	movq	-576(%rbp), %rdx
	xorl	%eax, %eax
.LVL3062:
	.p2align 4,,10
	.p2align 3
.L1789:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3063:
	addq	$4, %rdx
	cmpl	%eax, -636(%rbp)
	jg	.L1789
.LVL3064:
.L1788:
.LBE13198:
.LBE13196:
.LBE13195:
	.loc 2 277 0
	movq	-568(%rbp), %rdi
	addq	$80, %r12
.LVL3065:
.LBB13200:
.LBB13199:
	.loc 2 371 0
	movq	$0, -616(%rbp)
.LVL3066:
.LBE13199:
.LBE13200:
	.loc 2 277 0
	cmpq	%r12, %rdi
	je	.L1714
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3067:
.L1714:
.LBE13194:
.LBE13193:
	.loc 1 609 0
	movq	-56(%rbp), %rax
	xorq	%fs:40, %rax
	jne	.L1891
	leaq	-40(%rbp), %rsp
	popq	%rbx
.LVL3068:
	popq	%r12
	popq	%r13
	popq	%r14
.LVL3069:
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
.LVL3070:
	ret
.LVL3071:
	.p2align 4,,10
	.p2align 3
.L1884:
	.cfi_restore_state
.LBB13201:
.LBB13038:
	.loc 2 291 0
	movl	%eax, -636(%rbp)
	.loc 2 292 0
	movl	-344(%rbp), %eax
	movq	-280(%rbp), %rdx
	movl	%eax, -632(%rbp)
	.loc 2 293 0
	movl	-340(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -628(%rbp)
	movq	-568(%rbp), %rax
.LVL3072:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL3073:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L1723
.LVL3074:
	.p2align 4,,10
	.p2align 3
.L1887:
.LBE13038:
.LBE13201:
.LBB13202:
.LBB13098:
	.loc 2 291 0
	movl	%eax, -540(%rbp)
	.loc 2 292 0
	movl	-248(%rbp), %eax
	movq	-184(%rbp), %rdx
	movl	%eax, -536(%rbp)
	.loc 2 293 0
	movl	-244(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -532(%rbp)
	movq	-472(%rbp), %rax
.LVL3075:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL3076:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L1736
.LVL3077:
	.p2align 4,,10
	.p2align 3
.L1890:
.LBE13098:
.LBE13202:
.LBB13203:
.LBB13158:
	.loc 2 291 0
	movl	%eax, -444(%rbp)
	.loc 2 292 0
	movl	-152(%rbp), %eax
	movq	-88(%rbp), %rdx
	movl	%eax, -440(%rbp)
	.loc 2 293 0
	movl	-148(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -436(%rbp)
	movq	-376(%rbp), %rax
.LVL3078:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL3079:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L1749
.LVL3080:
	.p2align 4,,10
	.p2align 3
.L1882:
.LBE13158:
.LBE13203:
.LBB13204:
.LBB13039:
.LBB13035:
.LBB13032:
	.loc 2 367 0
	movq	%r12, %rdi
.LEHB62:
	call	_ZN2cv3Mat10deallocateEv
.LVL3081:
.LEHE62:
	jmp	.L1795
.LVL3082:
	.p2align 4,,10
	.p2align 3
.L1885:
.LBE13032:
.LBE13035:
.LBE13039:
.LBE13204:
.LBB13205:
.LBB13099:
.LBB13095:
.LBB13092:
	movq	%r13, %rdi
.LEHB63:
	call	_ZN2cv3Mat10deallocateEv
.LVL3083:
.LEHE63:
	jmp	.L1799
.LVL3084:
	.p2align 4,,10
	.p2align 3
.L1888:
.LBE13092:
.LBE13095:
.LBE13099:
.LBE13205:
.LBB13206:
.LBB13159:
.LBB13155:
.LBB13152:
	movq	%r15, %rdi
.LEHB64:
	call	_ZN2cv3Mat10deallocateEv
.LVL3085:
.LEHE64:
	jmp	.L1803
.LVL3086:
	.p2align 4,,10
	.p2align 3
.L1760:
.LBE13152:
.LBE13155:
.LBE13159:
.LBE13206:
.LBB13207:
	.loc 1 386 0
	xorl	%ecx, %ecx
.LVL3087:
	.p2align 4,,10
	.p2align 3
.L1769:
	.loc 1 387 0
	movzbl	(%rdx,%rcx), %esi
	pxor	%xmm0, %xmm0
	cvtsi2ss	%esi, %xmm0
	movss	%xmm0, (%rax,%rcx,4)
.LVL3088:
	addq	$1, %rcx
.LVL3089:
	.loc 1 386 0
	cmpl	%ecx, %ebx
	jg	.L1769
	jmp	.L1767
.LVL3090:
.L1889:
.LBE13207:
.LBB13208:
.LBB13160:
	.loc 2 288 0
	movl	-160(%rbp), %eax
.LBB13156:
.LBB13153:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL3091:
.LBE13153:
.LBE13156:
	.loc 2 288 0
	movl	%eax, -448(%rbp)
	jmp	.L1812
.LVL3092:
.L1886:
.LBE13160:
.LBE13208:
.LBB13209:
.LBB13100:
	movl	-256(%rbp), %eax
.LBB13096:
.LBB13093:
	.loc 2 371 0
	movq	$0, -520(%rbp)
.LVL3093:
.LBE13093:
.LBE13096:
	.loc 2 288 0
	movl	%eax, -544(%rbp)
	jmp	.L1811
.LVL3094:
.L1883:
.LBE13100:
.LBE13209:
.LBB13210:
.LBB13040:
	movl	-352(%rbp), %eax
.LBB13036:
.LBB13033:
	.loc 2 371 0
	movq	$0, -616(%rbp)
.LVL3095:
.LBE13033:
.LBE13036:
	.loc 2 288 0
	movl	%eax, -640(%rbp)
	jmp	.L1810
.LVL3096:
.L1891:
.LBE13040:
.LBE13210:
	.loc 1 609 0
	call	__stack_chk_fail
.LVL3097:
.L1820:
	movq	%rax, %rbx
.LVL3098:
	jmp	.L1793
.LVL3099:
.L1817:
	movq	%rax, %rbx
.LVL3100:
	jmp	.L1791
.LVL3101:
.L1793:
	.loc 1 366 0 discriminator 2
	leaq	-160(%rbp), %rdi
.LVL3102:
	call	_ZN2cv3MatD1Ev
.LVL3103:
.L1791:
	.loc 1 362 0
	movq	%r15, %rdi
	call	_ZN2cv3MatD1Ev
.LVL3104:
	.loc 1 361 0
	movq	%r13, %rdi
	call	_ZN2cv3MatD1Ev
.LVL3105:
	.loc 1 360 0
	movq	%r12, %rdi
	call	_ZN2cv3MatD1Ev
.LVL3106:
	movq	%rbx, %rdi
.LEHB65:
	call	_Unwind_Resume
.LVL3107:
.LEHE65:
.L1818:
	movq	%rax, %rbx
.LVL3108:
	jmp	.L1790
.LVL3109:
.L1819:
	movq	%rax, %rbx
.LVL3110:
	jmp	.L1792
.LVL3111:
.L1790:
	.loc 1 364 0 discriminator 2
	leaq	-352(%rbp), %rdi
.LVL3112:
	call	_ZN2cv3MatD1Ev
.LVL3113:
	jmp	.L1791
.LVL3114:
.L1792:
	.loc 1 365 0 discriminator 2
	leaq	-256(%rbp), %rdi
.LVL3115:
	call	_ZN2cv3MatD1Ev
.LVL3116:
	jmp	.L1791
	.cfi_endproc
.LFE11599:
	.section	.gcc_except_table
.LLSDA11599:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE11599-.LLSDACSB11599
.LLSDACSB11599:
	.uleb128 .LEHB56-.LFB11599
	.uleb128 .LEHE56-.LEHB56
	.uleb128 .L1817-.LFB11599
	.uleb128 0
	.uleb128 .LEHB57-.LFB11599
	.uleb128 .LEHE57-.LEHB57
	.uleb128 .L1818-.LFB11599
	.uleb128 0
	.uleb128 .LEHB58-.LFB11599
	.uleb128 .LEHE58-.LEHB58
	.uleb128 .L1817-.LFB11599
	.uleb128 0
	.uleb128 .LEHB59-.LFB11599
	.uleb128 .LEHE59-.LEHB59
	.uleb128 .L1819-.LFB11599
	.uleb128 0
	.uleb128 .LEHB60-.LFB11599
	.uleb128 .LEHE60-.LEHB60
	.uleb128 .L1817-.LFB11599
	.uleb128 0
	.uleb128 .LEHB61-.LFB11599
	.uleb128 .LEHE61-.LEHB61
	.uleb128 .L1820-.LFB11599
	.uleb128 0
	.uleb128 .LEHB62-.LFB11599
	.uleb128 .LEHE62-.LEHB62
	.uleb128 .L1818-.LFB11599
	.uleb128 0
	.uleb128 .LEHB63-.LFB11599
	.uleb128 .LEHE63-.LEHB63
	.uleb128 .L1819-.LFB11599
	.uleb128 0
	.uleb128 .LEHB64-.LFB11599
	.uleb128 .LEHE64-.LEHB64
	.uleb128 .L1820-.LFB11599
	.uleb128 0
	.uleb128 .LEHB65-.LFB11599
	.uleb128 .LEHE65-.LEHB65
	.uleb128 0
	.uleb128 0
.LLSDACSE11599:
	.section	.text._ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_,comdat
	.size	_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_, .-_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_
	.section	.text.unlikely._ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_,comdat
.LCOLDE77:
	.section	.text._ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_,comdat
.LHOTE77:
	.section	.text.unlikely._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_,comdat
	.align 2
.LCOLDB78:
	.section	.text._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_,comdat
.LHOTB78:
	.align 2
	.p2align 4,,15
	.weak	_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_
	.type	_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_, @function
_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_:
.LFB11600:
	.loc 1 354 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA11600
.LVL3117:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdx, %rax
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$776, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	.loc 1 354 0
	movq	%rdi, -800(%rbp)
	movq	%rdx, -808(%rbp)
	movq	%fs:40, %rbx
	movq	%rbx, -56(%rbp)
	xorl	%ebx, %ebx
	.loc 1 356 0
	movl	8(%rsi), %ebx
	.loc 1 359 0
	movl	(%rsi), %edx
.LVL3118:
	.loc 1 356 0
	movl	%ebx, -788(%rbp)
.LVL3119:
	.loc 1 357 0
	movl	12(%rsi), %ebx
.LVL3120:
	.loc 1 359 0
	andl	$4095, %edx
	.loc 1 358 0
	leal	8(%rbx), %edi
.LVL3121:
	movl	%edi, -792(%rbp)
.LVL3122:
	.loc 1 359 0
	movq	%rax, %rdi
.LVL3123:
	movl	(%rax), %eax
.LVL3124:
	movl	%eax, -816(%rbp)
	andl	$4095, %eax
	cmpl	%eax, %edx
	jne	.L1893
.LBB13333:
.LBB13334:
	.loc 2 713 0 discriminator 2
	movq	64(%rdi), %rax
.LBE13334:
.LBE13333:
.LBB13335:
.LBB13336:
	movq	64(%rsi), %rdx
	movq	%rsi, %r14
.LVL3125:
.LBE13336:
.LBE13335:
.LBB13337:
.LBB13338:
	.loc 13 1893 0 discriminator 2
	movl	(%rax), %edi
.LVL3126:
	cmpl	%edi, (%rdx)
	jne	.L1893
	movl	4(%rax), %eax
	cmpl	%eax, 4(%rdx)
	je	.L1894
.LVL3127:
.L1893:
.LBE13338:
.LBE13337:
	.loc 1 359 0 discriminator 7
	movl	$_ZZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_E19__PRETTY_FUNCTION__, %ecx
	movl	$359, %edx
	movl	$.LC45, %esi
.LVL3128:
	movl	$.LC76, %edi
	call	__assert_fail
.LVL3129:
	.p2align 4,,10
	.p2align 3
.L1894:
.LBB13339:
.LBB13340:
.LBB13341:
	.loc 2 709 0
	leaq	-640(%rbp), %r12
.LVL3130:
.LBE13341:
.LBE13340:
.LBE13339:
.LBB13362:
.LBB13363:
.LBB13364:
	leaq	-544(%rbp), %r13
.LBE13364:
.LBE13363:
.LBE13362:
.LBB13383:
.LBB13384:
.LBB13385:
	leaq	-448(%rbp), %r15
.LBE13385:
.LBE13384:
.LBE13383:
.LBB13404:
.LBB13344:
.LBB13345:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE13345:
.LBE13344:
.LBB13348:
.LBB13349:
	.loc 2 62 0
	movq	$0, -592(%rbp)
.LBE13349:
.LBE13348:
.LBB13354:
.LBB13342:
	.loc 2 709 0
	leaq	8(%r12), %rax
.LBE13342:
.LBE13354:
.LBB13355:
.LBB13350:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm5
	.loc 2 62 0
	movq	$0, -600(%rbp)
	movq	$0, -608(%rbp)
.LBE13350:
.LBE13355:
.LBE13404:
.LBB13405:
.LBB13406:
.LBB13407:
	.loc 2 353 0
	leaq	-688(%rbp), %rdx
.LBE13407:
.LBE13406:
.LBE13405:
.LBB13426:
.LBB13356:
.LBB13343:
	.loc 2 709 0
	movq	%rax, -576(%rbp)
.LVL3131:
.LBE13343:
.LBE13356:
.LBB13357:
.LBB13346:
	.loc 2 738 0
	leaq	80(%r12), %rax
.LBE13346:
.LBE13357:
.LBB13358:
.LBB13351:
	.loc 2 62 0
	movq	$0, -624(%rbp)
.LBE13351:
.LBE13358:
.LBB13359:
.LBB13347:
	.loc 2 738 0
	movaps	%xmm0, -560(%rbp)
.LVL3132:
	movq	%rax, -568(%rbp)
.LBE13347:
.LBE13359:
.LBE13426:
.LBB13427:
.LBB13367:
.LBB13365:
	.loc 2 709 0
	leaq	8(%r13), %rax
.LBE13365:
.LBE13367:
.LBE13427:
.LBB13428:
.LBB13360:
.LBB13352:
	.loc 2 63 0
	movq	$0, -616(%rbp)
	.loc 2 60 0
	movaps	%xmm5, -640(%rbp)
.LBE13352:
.LBE13360:
.LBE13428:
.LBB13429:
.LBB13411:
.LBB13408:
	.loc 2 353 0
	leaq	-352(%rbp), %rdi
	movl	$13, %ecx
	movl	$2, %esi
.LVL3133:
.LBE13408:
.LBE13411:
.LBE13429:
.LBB13430:
.LBB13368:
.LBB13369:
	.loc 2 738 0
	movaps	%xmm0, -464(%rbp)
.LBE13369:
.LBE13368:
.LBB13372:
.LBB13373:
	.loc 2 60 0
	movaps	%xmm5, -544(%rbp)
.LBE13373:
.LBE13372:
.LBE13430:
.LBB13431:
.LBB13388:
.LBB13389:
	.loc 2 738 0
	movaps	%xmm0, -368(%rbp)
.LBE13389:
.LBE13388:
.LBB13392:
.LBB13393:
	.loc 2 60 0
	movaps	%xmm5, -448(%rbp)
.LBE13393:
.LBE13392:
.LBE13431:
.LBB13432:
.LBB13377:
.LBB13366:
	.loc 2 709 0
	movq	%rax, -480(%rbp)
.LBE13366:
.LBE13377:
.LBB13378:
.LBB13370:
	.loc 2 738 0
	leaq	80(%r13), %rax
.LBE13370:
.LBE13378:
.LBE13432:
.LBB13433:
.LBB13361:
.LBB13353:
	.loc 2 64 0
	movq	$0, -584(%rbp)
.LVL3134:
.LBE13353:
.LBE13361:
.LBE13433:
.LBB13434:
.LBB13379:
.LBB13374:
	.loc 2 62 0
	movq	$0, -496(%rbp)
	movq	$0, -504(%rbp)
.LBE13374:
.LBE13379:
.LBB13380:
.LBB13371:
	.loc 2 738 0
	movq	%rax, -472(%rbp)
.LBE13371:
.LBE13380:
.LBE13434:
.LBB13435:
.LBB13397:
.LBB13386:
	.loc 2 709 0
	leaq	8(%r15), %rax
.LBE13386:
.LBE13397:
.LBE13435:
.LBB13436:
.LBB13381:
.LBB13375:
	.loc 2 62 0
	movq	$0, -512(%rbp)
	movq	$0, -528(%rbp)
	.loc 2 63 0
	movq	$0, -520(%rbp)
.LBE13375:
.LBE13381:
.LBE13436:
.LBB13437:
.LBB13398:
.LBB13387:
	.loc 2 709 0
	movq	%rax, -384(%rbp)
.LBE13387:
.LBE13398:
.LBB13399:
.LBB13390:
	.loc 2 738 0
	leaq	80(%r15), %rax
.LBE13390:
.LBE13399:
.LBE13437:
.LBB13438:
.LBB13382:
.LBB13376:
	.loc 2 64 0
	movq	$0, -488(%rbp)
.LVL3135:
.LBE13376:
.LBE13382:
.LBE13438:
.LBB13439:
.LBB13400:
.LBB13394:
	.loc 2 62 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
.LBE13394:
.LBE13400:
.LBB13401:
.LBB13391:
	.loc 2 738 0
	movq	%rax, -376(%rbp)
.LBE13391:
.LBE13401:
.LBE13439:
.LBB13440:
.LBB13412:
.LBB13413:
	.loc 2 709 0
	leaq	-352(%rbp), %rax
.LBE13413:
.LBE13412:
.LBE13440:
.LBB13441:
.LBB13402:
.LBB13395:
	.loc 2 62 0
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
	.loc 2 63 0
	movq	$0, -424(%rbp)
.LBE13395:
.LBE13402:
.LBE13441:
.LBB13442:
.LBB13416:
.LBB13414:
	.loc 2 709 0
	addq	$8, %rax
.LBE13414:
.LBE13416:
.LBE13442:
.LBB13443:
.LBB13403:
.LBB13396:
	.loc 2 64 0
	movq	$0, -392(%rbp)
.LVL3136:
.LBE13396:
.LBE13403:
.LBE13443:
.LBB13444:
.LBB13417:
.LBB13415:
	.loc 2 709 0
	movq	%rax, -288(%rbp)
.LVL3137:
.LBE13415:
.LBE13417:
.LBB13418:
.LBB13419:
	.loc 2 738 0
	leaq	-352(%rbp), %rax
	addq	$80, %rax
	movq	%rax, -280(%rbp)
.LBE13419:
.LBE13418:
.LBB13421:
.LBB13409:
	.loc 2 352 0
	movl	-788(%rbp), %eax
.LBE13409:
.LBE13421:
.LBB13422:
.LBB13420:
	.loc 2 738 0
	movaps	%xmm0, -272(%rbp)
.LVL3138:
.LBE13420:
.LBE13422:
.LBB13423:
.LBB13424:
	.loc 2 62 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
	.loc 2 60 0
	movaps	%xmm5, -352(%rbp)
	.loc 2 62 0
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
	.loc 2 63 0
	movq	$0, -328(%rbp)
	.loc 2 64 0
	movq	$0, -296(%rbp)
.LVL3139:
.LBE13424:
.LBE13423:
.LBB13425:
.LBB13410:
	.loc 2 352 0
	movl	%eax, -688(%rbp)
	movl	%ebx, -684(%rbp)
.LEHB66:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL3140:
.LEHE66:
.LBE13410:
.LBE13425:
.LBE13444:
.LBB13445:
.LBB13446:
	.loc 2 285 0
	movq	-328(%rbp), %rax
	testq	%rax, %rax
	je	.L1896
	.loc 2 286 0
	lock addl	$1, (%rax)
.L1896:
.LVL3141:
.LBB13447:
.LBB13448:
	.loc 2 366 0
	movq	-616(%rbp), %rax
	testq	%rax, %rax
	je	.L1973
	lock subl	$1, (%rax)
	je	.L2060
.L1973:
.LBB13449:
	.loc 2 369 0
	movl	-636(%rbp), %ecx
.LBE13449:
	.loc 2 368 0
	movq	$0, -592(%rbp)
	movq	$0, -600(%rbp)
	movq	$0, -608(%rbp)
	movq	$0, -624(%rbp)
.LVL3142:
.LBB13450:
	.loc 2 369 0
	testl	%ecx, %ecx
	jle	.L2061
	movq	-576(%rbp), %rdx
	xorl	%eax, %eax
.LVL3143:
	.p2align 4,,10
	.p2align 3
.L1899:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-636(%rbp), %ecx
	addl	$1, %eax
.LVL3144:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L1899
.LBE13450:
.LBE13448:
.LBE13447:
	.loc 2 288 0
	movl	-352(%rbp), %eax
.LVL3145:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB13454:
.LBB13451:
	.loc 2 371 0
	movq	$0, -616(%rbp)
.LVL3146:
.LBE13451:
.LBE13454:
	.loc 2 288 0
	movl	%eax, -640(%rbp)
	.loc 2 289 0
	jg	.L1900
.L1988:
	movl	-348(%rbp), %eax
	cmpl	$2, %eax
	jle	.L2062
.L1900:
	.loc 2 298 0
	leaq	-352(%rbp), %rsi
.LVL3147:
	movq	%r12, %rdi
.LEHB67:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL3148:
.LEHE67:
.L1901:
	.loc 2 299 0
	movdqa	-336(%rbp), %xmm0
	.loc 2 303 0
	movq	-328(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -624(%rbp)
.LBE13446:
.LBE13445:
.LBB13461:
.LBB13462:
.LBB13463:
.LBB13464:
	.loc 2 366 0
	testq	%rax, %rax
.LBE13464:
.LBE13463:
.LBE13462:
.LBE13461:
.LBB13472:
.LBB13457:
	.loc 2 299 0
	movdqa	-320(%rbp), %xmm0
	movaps	%xmm0, -608(%rbp)
	movdqa	-304(%rbp), %xmm0
	movaps	%xmm0, -592(%rbp)
.LVL3149:
.LBE13457:
.LBE13472:
.LBB13473:
.LBB13471:
.LBB13469:
.LBB13467:
	.loc 2 366 0
	je	.L1976
	lock subl	$1, (%rax)
	jne	.L1976
	.loc 2 367 0
	leaq	-352(%rbp), %rdi
.LVL3150:
	call	_ZN2cv3Mat10deallocateEv
.LVL3151:
.L1976:
.LBB13465:
	.loc 2 369 0
	movl	-348(%rbp), %r11d
.LBE13465:
	.loc 2 368 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
.LVL3152:
.LBB13466:
	.loc 2 369 0
	testl	%r11d, %r11d
	jle	.L1907
	movq	-288(%rbp), %rdx
	xorl	%eax, %eax
.LVL3153:
	.p2align 4,,10
	.p2align 3
.L1908:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3154:
	addq	$4, %rdx
	cmpl	%eax, -348(%rbp)
	jg	.L1908
.LVL3155:
.L1907:
.LBE13466:
.LBE13467:
.LBE13469:
	.loc 2 277 0
	movq	-280(%rbp), %rdi
	leaq	-352(%rbp), %rax
.LVL3156:
.LBB13470:
.LBB13468:
	.loc 2 371 0
	movq	$0, -328(%rbp)
.LVL3157:
.LBE13468:
.LBE13470:
	.loc 2 277 0
	addq	$80, %rax
.LVL3158:
	cmpq	%rax, %rdi
	je	.L1906
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3159:
.L1906:
.LBE13471:
.LBE13473:
.LBB13474:
.LBB13475:
.LBB13476:
	.loc 2 709 0
	leaq	-256(%rbp), %rax
.LVL3160:
.LBE13476:
.LBE13475:
.LBB13479:
.LBB13480:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE13480:
.LBE13479:
.LBB13484:
.LBB13485:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm3
.LBE13485:
.LBE13484:
.LBB13488:
.LBB13489:
	.loc 2 353 0
	leaq	-672(%rbp), %rdx
.LBE13489:
.LBE13488:
.LBB13494:
.LBB13477:
	.loc 2 709 0
	addq	$8, %rax
.LVL3161:
.LBE13477:
.LBE13494:
.LBB13495:
.LBB13490:
	.loc 2 353 0
	leaq	-256(%rbp), %rdi
.LVL3162:
	movl	$13, %ecx
.LBE13490:
.LBE13495:
.LBB13496:
.LBB13478:
	.loc 2 709 0
	movq	%rax, -192(%rbp)
.LVL3163:
.LBE13478:
.LBE13496:
.LBB13497:
.LBB13481:
	.loc 2 738 0
	leaq	-256(%rbp), %rax
.LBE13481:
.LBE13497:
.LBB13498:
.LBB13491:
	.loc 2 353 0
	movl	$2, %esi
.LBE13491:
.LBE13498:
.LBB13499:
.LBB13482:
	.loc 2 738 0
	movaps	%xmm0, -176(%rbp)
.LVL3164:
	addq	$80, %rax
.LBE13482:
.LBE13499:
.LBB13500:
.LBB13486:
	.loc 2 62 0
	movq	$0, -208(%rbp)
	movq	$0, -216(%rbp)
	.loc 2 60 0
	movaps	%xmm3, -256(%rbp)
.LBE13486:
.LBE13500:
.LBB13501:
.LBB13483:
	.loc 2 738 0
	movq	%rax, -184(%rbp)
.LBE13483:
.LBE13501:
.LBB13502:
.LBB13492:
	.loc 2 352 0
	movl	-788(%rbp), %eax
.LBE13492:
.LBE13502:
.LBB13503:
.LBB13487:
	.loc 2 62 0
	movq	$0, -224(%rbp)
	movq	$0, -240(%rbp)
	.loc 2 63 0
	movq	$0, -232(%rbp)
	.loc 2 64 0
	movq	$0, -200(%rbp)
.LVL3165:
.LBE13487:
.LBE13503:
.LBB13504:
.LBB13493:
	.loc 2 352 0
	movl	%eax, -672(%rbp)
	movl	%ebx, -668(%rbp)
.LEHB68:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL3166:
.LEHE68:
.LBE13493:
.LBE13504:
.LBE13474:
.LBB13505:
.LBB13506:
	.loc 2 285 0
	movq	-232(%rbp), %rax
	testq	%rax, %rax
	je	.L1909
	.loc 2 286 0
	lock addl	$1, (%rax)
.L1909:
.LVL3167:
.LBB13507:
.LBB13508:
	.loc 2 366 0
	movq	-520(%rbp), %rax
	testq	%rax, %rax
	je	.L1977
	lock subl	$1, (%rax)
	je	.L2063
.L1977:
.LBB13509:
	.loc 2 369 0
	movl	-540(%rbp), %edx
.LBE13509:
	.loc 2 368 0
	movq	$0, -496(%rbp)
	movq	$0, -504(%rbp)
	movq	$0, -512(%rbp)
	movq	$0, -528(%rbp)
.LVL3168:
.LBB13510:
	.loc 2 369 0
	testl	%edx, %edx
	jle	.L2064
	movq	-480(%rbp), %rdx
	xorl	%eax, %eax
.LVL3169:
	.p2align 4,,10
	.p2align 3
.L1912:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-540(%rbp), %ecx
	addl	$1, %eax
.LVL3170:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L1912
.LBE13510:
.LBE13508:
.LBE13507:
	.loc 2 288 0
	movl	-256(%rbp), %eax
.LVL3171:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB13514:
.LBB13511:
	.loc 2 371 0
	movq	$0, -520(%rbp)
.LVL3172:
.LBE13511:
.LBE13514:
	.loc 2 288 0
	movl	%eax, -544(%rbp)
	.loc 2 289 0
	jg	.L1913
.L1989:
	movl	-252(%rbp), %eax
	cmpl	$2, %eax
	jle	.L2065
.L1913:
	.loc 2 298 0
	leaq	-256(%rbp), %rsi
.LVL3173:
	movq	%r13, %rdi
.LEHB69:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL3174:
.LEHE69:
.L1914:
	.loc 2 299 0
	movdqa	-240(%rbp), %xmm0
	.loc 2 303 0
	movq	-232(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -528(%rbp)
.LBE13506:
.LBE13505:
.LBB13521:
.LBB13522:
.LBB13523:
.LBB13524:
	.loc 2 366 0
	testq	%rax, %rax
.LBE13524:
.LBE13523:
.LBE13522:
.LBE13521:
.LBB13532:
.LBB13517:
	.loc 2 299 0
	movdqa	-224(%rbp), %xmm0
	movaps	%xmm0, -512(%rbp)
	movdqa	-208(%rbp), %xmm0
	movaps	%xmm0, -496(%rbp)
.LVL3175:
.LBE13517:
.LBE13532:
.LBB13533:
.LBB13531:
.LBB13529:
.LBB13527:
	.loc 2 366 0
	je	.L1980
	lock subl	$1, (%rax)
	jne	.L1980
	.loc 2 367 0
	leaq	-256(%rbp), %rdi
.LVL3176:
	call	_ZN2cv3Mat10deallocateEv
.LVL3177:
.L1980:
.LBB13525:
	.loc 2 369 0
	movl	-252(%rbp), %r10d
.LBE13525:
	.loc 2 368 0
	movq	$0, -208(%rbp)
	movq	$0, -216(%rbp)
	movq	$0, -224(%rbp)
	movq	$0, -240(%rbp)
.LVL3178:
.LBB13526:
	.loc 2 369 0
	testl	%r10d, %r10d
	jle	.L1920
	movq	-192(%rbp), %rdx
	xorl	%eax, %eax
.LVL3179:
	.p2align 4,,10
	.p2align 3
.L1921:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3180:
	addq	$4, %rdx
	cmpl	%eax, -252(%rbp)
	jg	.L1921
.LVL3181:
.L1920:
.LBE13526:
.LBE13527:
.LBE13529:
	.loc 2 277 0
	movq	-184(%rbp), %rdi
	leaq	-256(%rbp), %rax
.LVL3182:
.LBB13530:
.LBB13528:
	.loc 2 371 0
	movq	$0, -232(%rbp)
.LVL3183:
.LBE13528:
.LBE13530:
	.loc 2 277 0
	addq	$80, %rax
.LVL3184:
	cmpq	%rax, %rdi
	je	.L1919
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3185:
.L1919:
.LBE13531:
.LBE13533:
.LBB13534:
.LBB13535:
.LBB13536:
	.loc 2 709 0
	leaq	-160(%rbp), %rax
.LVL3186:
.LBE13536:
.LBE13535:
.LBB13539:
.LBB13540:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE13540:
.LBE13539:
.LBB13544:
.LBB13545:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm4
.LBE13545:
.LBE13544:
.LBB13548:
.LBB13549:
	.loc 2 353 0
	leaq	-656(%rbp), %rdx
.LBE13549:
.LBE13548:
.LBB13554:
.LBB13537:
	.loc 2 709 0
	addq	$8, %rax
.LVL3187:
.LBE13537:
.LBE13554:
.LBB13555:
.LBB13550:
	.loc 2 353 0
	leaq	-160(%rbp), %rdi
.LVL3188:
	movl	$13, %ecx
.LBE13550:
.LBE13555:
.LBB13556:
.LBB13538:
	.loc 2 709 0
	movq	%rax, -96(%rbp)
.LVL3189:
.LBE13538:
.LBE13556:
.LBB13557:
.LBB13541:
	.loc 2 738 0
	leaq	-160(%rbp), %rax
.LBE13541:
.LBE13557:
.LBB13558:
.LBB13551:
	.loc 2 353 0
	movl	$2, %esi
.LBE13551:
.LBE13558:
.LBB13559:
.LBB13542:
	.loc 2 738 0
	movaps	%xmm0, -80(%rbp)
.LVL3190:
	addq	$80, %rax
.LBE13542:
.LBE13559:
.LBB13560:
.LBB13546:
	.loc 2 62 0
	movq	$0, -112(%rbp)
	movq	$0, -120(%rbp)
	.loc 2 60 0
	movaps	%xmm4, -160(%rbp)
.LBE13546:
.LBE13560:
.LBB13561:
.LBB13543:
	.loc 2 738 0
	movq	%rax, -88(%rbp)
.LBE13543:
.LBE13561:
.LBB13562:
.LBB13552:
	.loc 2 352 0
	movl	-792(%rbp), %eax
.LBE13552:
.LBE13562:
.LBB13563:
.LBB13547:
	.loc 2 62 0
	movq	$0, -128(%rbp)
	movq	$0, -144(%rbp)
	.loc 2 63 0
	movq	$0, -136(%rbp)
	.loc 2 64 0
	movq	$0, -104(%rbp)
.LVL3191:
.LBE13547:
.LBE13563:
.LBB13564:
.LBB13553:
	.loc 2 352 0
	movl	$4, -656(%rbp)
	movl	%eax, -652(%rbp)
.LEHB70:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL3192:
.LEHE70:
.LBE13553:
.LBE13564:
.LBE13534:
.LBB13565:
.LBB13566:
	.loc 2 285 0
	movq	-136(%rbp), %rax
	testq	%rax, %rax
	je	.L1922
	.loc 2 286 0
	lock addl	$1, (%rax)
.L1922:
.LVL3193:
.LBB13567:
.LBB13568:
	.loc 2 366 0
	movq	-424(%rbp), %rax
	testq	%rax, %rax
	je	.L1981
	lock subl	$1, (%rax)
	je	.L2066
.L1981:
.LBB13569:
	.loc 2 369 0
	movl	-444(%rbp), %eax
.LBE13569:
	.loc 2 368 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
.LVL3194:
.LBB13570:
	.loc 2 369 0
	testl	%eax, %eax
	jle	.L2067
	movq	-384(%rbp), %rdx
	xorl	%eax, %eax
.LVL3195:
	.p2align 4,,10
	.p2align 3
.L1925:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-444(%rbp), %ecx
	addl	$1, %eax
.LVL3196:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L1925
.LBE13570:
.LBE13568:
.LBE13567:
	.loc 2 288 0
	movl	-160(%rbp), %eax
.LVL3197:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB13574:
.LBB13571:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL3198:
.LBE13571:
.LBE13574:
	.loc 2 288 0
	movl	%eax, -448(%rbp)
	.loc 2 289 0
	jg	.L1926
.L1990:
	movl	-156(%rbp), %eax
	cmpl	$2, %eax
	jle	.L2068
.L1926:
	.loc 2 298 0
	leaq	-160(%rbp), %rsi
.LVL3199:
	movq	%r15, %rdi
.LEHB71:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL3200:
.LEHE71:
.L1927:
	.loc 2 299 0
	movdqa	-144(%rbp), %xmm0
	.loc 2 303 0
	movq	-136(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -432(%rbp)
.LBE13566:
.LBE13565:
.LBB13581:
.LBB13582:
.LBB13583:
.LBB13584:
	.loc 2 366 0
	testq	%rax, %rax
.LBE13584:
.LBE13583:
.LBE13582:
.LBE13581:
.LBB13592:
.LBB13577:
	.loc 2 299 0
	movdqa	-128(%rbp), %xmm0
	movaps	%xmm0, -416(%rbp)
	movdqa	-112(%rbp), %xmm0
	movaps	%xmm0, -400(%rbp)
.LVL3201:
.LBE13577:
.LBE13592:
.LBB13593:
.LBB13591:
.LBB13589:
.LBB13587:
	.loc 2 366 0
	je	.L1984
	lock subl	$1, (%rax)
	jne	.L1984
	.loc 2 367 0
	leaq	-160(%rbp), %rdi
.LVL3202:
	call	_ZN2cv3Mat10deallocateEv
.LVL3203:
.L1984:
.LBB13585:
	.loc 2 369 0
	movl	-156(%rbp), %r9d
.LBE13585:
	.loc 2 368 0
	movq	$0, -112(%rbp)
	movq	$0, -120(%rbp)
	movq	$0, -128(%rbp)
	movq	$0, -144(%rbp)
.LVL3204:
.LBB13586:
	.loc 2 369 0
	testl	%r9d, %r9d
	jle	.L1933
	movq	-96(%rbp), %rdx
	xorl	%eax, %eax
.LVL3205:
	.p2align 4,,10
	.p2align 3
.L1934:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3206:
	addq	$4, %rdx
	cmpl	%eax, -156(%rbp)
	jg	.L1934
.LVL3207:
.L1933:
.LBE13586:
.LBE13587:
.LBE13589:
	.loc 2 277 0
	movq	-88(%rbp), %rdi
	leaq	-160(%rbp), %rax
.LVL3208:
.LBB13590:
.LBB13588:
	.loc 2 371 0
	movq	$0, -136(%rbp)
.LVL3209:
.LBE13588:
.LBE13590:
	.loc 2 277 0
	addq	$80, %rax
.LVL3210:
	cmpq	%rax, %rdi
	je	.L1932
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3211:
.L1932:
.LBE13591:
.LBE13593:
	.loc 1 379 0
	addl	%ebx, %ebx
.LVL3212:
	movq	16(%r14), %rdx
	.loc 1 383 0
	movl	%ebx, %eax
	.loc 1 384 0
	movslq	%ebx, %rcx
	.loc 1 383 0
	sarl	$6, %eax
	.loc 1 384 0
	leaq	0(,%rcx,4), %rsi
	.loc 1 383 0
	sall	$4, %eax
	movl	%eax, -792(%rbp)
.LVL3213:
	.loc 1 384 0
	leaq	18(%rsi), %rax
.LVL3214:
	andq	$-16, %rax
	subq	%rax, %rsp
	leaq	3(%rsp), %rdi
	shrq	$2, %rdi
.LBB13594:
	.loc 1 386 0
	testl	%ebx, %ebx
.LBE13594:
	.loc 1 384 0
	leaq	0(,%rdi,4), %rax
.LVL3215:
.LBB13595:
	.loc 1 386 0
	jle	.L1945
	addq	%rax, %rsi
	cmpq	%rsi, %rdx
	setnb	%sil
	addq	%rdx, %rcx
	cmpq	%rcx, %rax
	setnb	%cl
	orb	%cl, %sil
	je	.L1938
	cmpl	$20, %ebx
	jbe	.L1938
	movq	%rax, %rsi
	andl	$15, %esi
	shrq	$2, %rsi
	negq	%rsi
	andl	$3, %esi
	cmpl	%ebx, %esi
	cmova	%ebx, %esi
	xorl	%ecx, %ecx
	testl	%esi, %esi
	je	.L1939
	.loc 1 387 0
	movzbl	(%rdx), %ecx
	pxor	%xmm0, %xmm0
	cmpl	$1, %esi
	cvtsi2ss	%ecx, %xmm0
	.loc 1 386 0
	movl	$1, %ecx
	.loc 1 387 0
	movss	%xmm0, 0(,%rdi,4)
.LVL3216:
	je	.L1939
	movzbl	1(%rdx), %ecx
	pxor	%xmm0, %xmm0
	cmpl	$2, %esi
	cvtsi2ss	%ecx, %xmm0
	.loc 1 386 0
	movl	$2, %ecx
	.loc 1 387 0
	movss	%xmm0, 4(,%rdi,4)
.LVL3217:
	je	.L1939
	movzbl	2(%rdx), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2ss	%ecx, %xmm0
	.loc 1 386 0
	movl	$3, %ecx
	.loc 1 387 0
	movss	%xmm0, 8(,%rdi,4)
.LVL3218:
.L1939:
	movl	%ebx, %r9d
	movl	%esi, %edi
	.loc 1 386 0
	xorl	%r10d, %r10d
	subl	%esi, %r9d
	leaq	(%rax,%rdi,4), %r8
	addq	%rdx, %rdi
	leal	-16(%r9), %esi
	shrl	$4, %esi
	addl	$1, %esi
	movl	%esi, %r11d
	sall	$4, %r11d
.L1942:
	.loc 1 387 0 discriminator 2
	movdqu	(%rdi), %xmm0
	addl	$1, %r10d
	addq	$64, %r8
	addq	$16, %rdi
	pmovzxbw	%xmm0, %xmm1
	psrldq	$8, %xmm0
	pmovzxbw	%xmm0, %xmm0
	pmovsxwd	%xmm1, %xmm2
	psrldq	$8, %xmm1
	pmovsxwd	%xmm1, %xmm1
	cvtdq2ps	%xmm2, %xmm2
	movaps	%xmm2, -64(%r8)
	cvtdq2ps	%xmm1, %xmm1
	movaps	%xmm1, -48(%r8)
	pmovsxwd	%xmm0, %xmm1
	psrldq	$8, %xmm0
	pmovsxwd	%xmm0, %xmm0
	cvtdq2ps	%xmm1, %xmm1
	movaps	%xmm1, -32(%r8)
	cvtdq2ps	%xmm0, %xmm0
	movaps	%xmm0, -16(%r8)
	cmpl	%r10d, %esi
	ja	.L1942
	addl	%r11d, %ecx
	cmpl	%r9d, %r11d
	je	.L1945
.LVL3219:
	.loc 1 387 0 is_stmt 0
	movslq	%ecx, %rdi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rdi), %esi
	cvtsi2ss	%esi, %xmm0
	.loc 1 386 0 is_stmt 1
	leal	1(%rcx), %esi
.LVL3220:
	cmpl	%esi, %ebx
	.loc 1 387 0
	movss	%xmm0, (%rax,%rdi,4)
	.loc 1 386 0
	jle	.L1945
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	2(%rcx), %esi
.LVL3221:
	cmpl	%esi, %ebx
	jle	.L1945
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	3(%rcx), %esi
.LVL3222:
	cmpl	%esi, %ebx
	jle	.L1945
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	4(%rcx), %esi
.LVL3223:
	cmpl	%esi, %ebx
	jle	.L1945
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	5(%rcx), %esi
.LVL3224:
	cmpl	%esi, %ebx
	jle	.L1945
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	6(%rcx), %esi
.LVL3225:
	cmpl	%esi, %ebx
	jle	.L1945
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	7(%rcx), %esi
.LVL3226:
	cmpl	%esi, %ebx
	jle	.L1945
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	8(%rcx), %esi
.LVL3227:
	cmpl	%esi, %ebx
	jle	.L1945
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	9(%rcx), %esi
.LVL3228:
	cmpl	%esi, %ebx
	jle	.L1945
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	10(%rcx), %esi
.LVL3229:
	cmpl	%esi, %ebx
	jle	.L1945
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	11(%rcx), %esi
.LVL3230:
	cmpl	%esi, %ebx
	jle	.L1945
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	12(%rcx), %esi
.LVL3231:
	cmpl	%esi, %ebx
	jle	.L1945
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
	.loc 1 386 0
	leal	13(%rcx), %esi
.LVL3232:
	cmpl	%esi, %ebx
	jle	.L1945
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	.loc 1 386 0
	addl	$14, %ecx
	cmpl	%ecx, %ebx
	.loc 1 387 0
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%rax,%rsi,4)
.LVL3233:
	.loc 1 386 0
	jle	.L1945
	.loc 1 387 0
	movslq	%ecx, %rcx
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rcx), %edx
	cvtsi2ss	%edx, %xmm0
	movss	%xmm0, (%rax,%rcx,4)
.L1945:
	movq	%rax, -816(%rbp)
.LBE13595:
	.loc 1 390 0
	call	omp_get_wtime
.LVL3234:
.LBB13596:
	.loc 1 392 0
	movq	-816(%rbp), %rax
	pxor	%xmm0, %xmm0
	leaq	-784(%rbp), %rsi
	xorl	%ecx, %ecx
	movl	$4, %edx
	movl	$_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_._omp_fn.4, %edi
	movq	%r14, -752(%rbp)
	movq	%r12, -736(%rbp)
	movq	%rax, -760(%rbp)
	movl	-792(%rbp), %eax
	movaps	%xmm0, -784(%rbp)
	movq	%r15, -728(%rbp)
	movq	%r13, -720(%rbp)
	movl	%eax, -704(%rbp)
	movq	-800(%rbp), %rax
	movl	%ebx, -708(%rbp)
	movq	%rax, -768(%rbp)
	movq	-808(%rbp), %rax
	movq	%rax, -744(%rbp)
	movl	-788(%rbp), %eax
	movl	%eax, -712(%rbp)
	call	GOMP_parallel
.LVL3235:
.LBE13596:
	.loc 1 608 0
	call	omp_get_wtime
.LVL3236:
.LBB13597:
.LBB13598:
.LBB13599:
.LBB13600:
	.loc 2 366 0
	movq	-424(%rbp), %rax
	testq	%rax, %rax
	je	.L1985
	lock subl	$1, (%rax)
	jne	.L1985
	.loc 2 367 0
	movq	%r15, %rdi
	call	_ZN2cv3Mat10deallocateEv
.LVL3237:
.L1985:
.LBB13601:
	.loc 2 369 0
	movl	-444(%rbp), %r8d
.LBE13601:
	.loc 2 368 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
.LVL3238:
.LBB13602:
	.loc 2 369 0
	testl	%r8d, %r8d
	jle	.L1952
	movq	-384(%rbp), %rdx
	xorl	%eax, %eax
.LVL3239:
	.p2align 4,,10
	.p2align 3
.L1953:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3240:
	addq	$4, %rdx
	cmpl	%eax, -444(%rbp)
	jg	.L1953
.LVL3241:
.L1952:
.LBE13602:
.LBE13600:
.LBE13599:
	.loc 2 277 0
	movq	-376(%rbp), %rdi
	leaq	80(%r15), %rax
.LBB13604:
.LBB13603:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL3242:
.LBE13603:
.LBE13604:
	.loc 2 277 0
	cmpq	%rax, %rdi
	je	.L1951
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3243:
.L1951:
.LBE13598:
.LBE13597:
.LBB13605:
.LBB13606:
.LBB13607:
.LBB13608:
	.loc 2 366 0
	movq	-520(%rbp), %rax
	testq	%rax, %rax
	je	.L1986
	lock subl	$1, (%rax)
	jne	.L1986
	.loc 2 367 0
	movq	%r13, %rdi
	call	_ZN2cv3Mat10deallocateEv
.LVL3244:
.L1986:
.LBB13609:
	.loc 2 369 0
	movl	-540(%rbp), %edi
.LBE13609:
	.loc 2 368 0
	movq	$0, -496(%rbp)
	movq	$0, -504(%rbp)
	movq	$0, -512(%rbp)
	movq	$0, -528(%rbp)
.LVL3245:
.LBB13610:
	.loc 2 369 0
	testl	%edi, %edi
	jle	.L1959
	movq	-480(%rbp), %rdx
	xorl	%eax, %eax
.LVL3246:
	.p2align 4,,10
	.p2align 3
.L1960:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3247:
	addq	$4, %rdx
	cmpl	%eax, -540(%rbp)
	jg	.L1960
.LVL3248:
.L1959:
.LBE13610:
.LBE13608:
.LBE13607:
	.loc 2 277 0
	movq	-472(%rbp), %rdi
	addq	$80, %r13
.LVL3249:
.LBB13612:
.LBB13611:
	.loc 2 371 0
	movq	$0, -520(%rbp)
.LVL3250:
.LBE13611:
.LBE13612:
	.loc 2 277 0
	cmpq	%r13, %rdi
	je	.L1958
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3251:
.L1958:
.LBE13606:
.LBE13605:
.LBB13613:
.LBB13614:
.LBB13615:
.LBB13616:
	.loc 2 366 0
	movq	-616(%rbp), %rax
	testq	%rax, %rax
	je	.L1987
	lock subl	$1, (%rax)
	jne	.L1987
	.loc 2 367 0
	movq	%r12, %rdi
	call	_ZN2cv3Mat10deallocateEv
.LVL3252:
.L1987:
.LBB13617:
	.loc 2 369 0
	movl	-636(%rbp), %esi
.LBE13617:
	.loc 2 368 0
	movq	$0, -592(%rbp)
	movq	$0, -600(%rbp)
	movq	$0, -608(%rbp)
	movq	$0, -624(%rbp)
.LVL3253:
.LBB13618:
	.loc 2 369 0
	testl	%esi, %esi
	jle	.L1966
	movq	-576(%rbp), %rdx
	xorl	%eax, %eax
.LVL3254:
	.p2align 4,,10
	.p2align 3
.L1967:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3255:
	addq	$4, %rdx
	cmpl	%eax, -636(%rbp)
	jg	.L1967
.LVL3256:
.L1966:
.LBE13618:
.LBE13616:
.LBE13615:
	.loc 2 277 0
	movq	-568(%rbp), %rdi
	addq	$80, %r12
.LVL3257:
.LBB13620:
.LBB13619:
	.loc 2 371 0
	movq	$0, -616(%rbp)
.LVL3258:
.LBE13619:
.LBE13620:
	.loc 2 277 0
	cmpq	%r12, %rdi
	je	.L1892
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3259:
.L1892:
.LBE13614:
.LBE13613:
	.loc 1 609 0
	movq	-56(%rbp), %rax
	xorq	%fs:40, %rax
	jne	.L2069
	leaq	-40(%rbp), %rsp
	popq	%rbx
.LVL3260:
	popq	%r12
	popq	%r13
	popq	%r14
.LVL3261:
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
.LVL3262:
	ret
.LVL3263:
	.p2align 4,,10
	.p2align 3
.L2062:
	.cfi_restore_state
.LBB13621:
.LBB13458:
	.loc 2 291 0
	movl	%eax, -636(%rbp)
	.loc 2 292 0
	movl	-344(%rbp), %eax
	movq	-280(%rbp), %rdx
	movl	%eax, -632(%rbp)
	.loc 2 293 0
	movl	-340(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -628(%rbp)
	movq	-568(%rbp), %rax
.LVL3264:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL3265:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L1901
.LVL3266:
	.p2align 4,,10
	.p2align 3
.L2065:
.LBE13458:
.LBE13621:
.LBB13622:
.LBB13518:
	.loc 2 291 0
	movl	%eax, -540(%rbp)
	.loc 2 292 0
	movl	-248(%rbp), %eax
	movq	-184(%rbp), %rdx
	movl	%eax, -536(%rbp)
	.loc 2 293 0
	movl	-244(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -532(%rbp)
	movq	-472(%rbp), %rax
.LVL3267:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL3268:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L1914
.LVL3269:
	.p2align 4,,10
	.p2align 3
.L2068:
.LBE13518:
.LBE13622:
.LBB13623:
.LBB13578:
	.loc 2 291 0
	movl	%eax, -444(%rbp)
	.loc 2 292 0
	movl	-152(%rbp), %eax
	movq	-88(%rbp), %rdx
	movl	%eax, -440(%rbp)
	.loc 2 293 0
	movl	-148(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -436(%rbp)
	movq	-376(%rbp), %rax
.LVL3270:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL3271:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L1927
.LVL3272:
	.p2align 4,,10
	.p2align 3
.L2060:
.LBE13578:
.LBE13623:
.LBB13624:
.LBB13459:
.LBB13455:
.LBB13452:
	.loc 2 367 0
	movq	%r12, %rdi
.LEHB72:
	call	_ZN2cv3Mat10deallocateEv
.LVL3273:
.LEHE72:
	jmp	.L1973
.LVL3274:
	.p2align 4,,10
	.p2align 3
.L2063:
.LBE13452:
.LBE13455:
.LBE13459:
.LBE13624:
.LBB13625:
.LBB13519:
.LBB13515:
.LBB13512:
	movq	%r13, %rdi
.LEHB73:
	call	_ZN2cv3Mat10deallocateEv
.LVL3275:
.LEHE73:
	jmp	.L1977
.LVL3276:
	.p2align 4,,10
	.p2align 3
.L2066:
.LBE13512:
.LBE13515:
.LBE13519:
.LBE13625:
.LBB13626:
.LBB13579:
.LBB13575:
.LBB13572:
	movq	%r15, %rdi
.LEHB74:
	call	_ZN2cv3Mat10deallocateEv
.LVL3277:
.LEHE74:
	jmp	.L1981
.LVL3278:
	.p2align 4,,10
	.p2align 3
.L1938:
.LBE13572:
.LBE13575:
.LBE13579:
.LBE13626:
.LBB13627:
	.loc 1 386 0
	xorl	%ecx, %ecx
.LVL3279:
	.p2align 4,,10
	.p2align 3
.L1947:
	.loc 1 387 0
	movzbl	(%rdx,%rcx), %esi
	pxor	%xmm0, %xmm0
	cvtsi2ss	%esi, %xmm0
	movss	%xmm0, (%rax,%rcx,4)
.LVL3280:
	addq	$1, %rcx
.LVL3281:
	.loc 1 386 0
	cmpl	%ecx, %ebx
	jg	.L1947
	jmp	.L1945
.LVL3282:
.L2067:
.LBE13627:
.LBB13628:
.LBB13580:
	.loc 2 288 0
	movl	-160(%rbp), %eax
.LBB13576:
.LBB13573:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL3283:
.LBE13573:
.LBE13576:
	.loc 2 288 0
	movl	%eax, -448(%rbp)
	jmp	.L1990
.LVL3284:
.L2064:
.LBE13580:
.LBE13628:
.LBB13629:
.LBB13520:
	movl	-256(%rbp), %eax
.LBB13516:
.LBB13513:
	.loc 2 371 0
	movq	$0, -520(%rbp)
.LVL3285:
.LBE13513:
.LBE13516:
	.loc 2 288 0
	movl	%eax, -544(%rbp)
	jmp	.L1989
.LVL3286:
.L2061:
.LBE13520:
.LBE13629:
.LBB13630:
.LBB13460:
	movl	-352(%rbp), %eax
.LBB13456:
.LBB13453:
	.loc 2 371 0
	movq	$0, -616(%rbp)
.LVL3287:
.LBE13453:
.LBE13456:
	.loc 2 288 0
	movl	%eax, -640(%rbp)
	jmp	.L1988
.LVL3288:
.L2069:
.LBE13460:
.LBE13630:
	.loc 1 609 0
	call	__stack_chk_fail
.LVL3289:
.L1998:
	movq	%rax, %rbx
.LVL3290:
	jmp	.L1971
.LVL3291:
.L1995:
	movq	%rax, %rbx
.LVL3292:
	jmp	.L1969
.LVL3293:
.L1971:
	.loc 1 371 0 discriminator 2
	leaq	-160(%rbp), %rdi
.LVL3294:
	call	_ZN2cv3MatD1Ev
.LVL3295:
.L1969:
	.loc 1 362 0
	movq	%r15, %rdi
	call	_ZN2cv3MatD1Ev
.LVL3296:
	.loc 1 361 0
	movq	%r13, %rdi
	call	_ZN2cv3MatD1Ev
.LVL3297:
	.loc 1 360 0
	movq	%r12, %rdi
	call	_ZN2cv3MatD1Ev
.LVL3298:
	movq	%rbx, %rdi
.LEHB75:
	call	_Unwind_Resume
.LVL3299:
.LEHE75:
.L1996:
	movq	%rax, %rbx
.LVL3300:
	jmp	.L1968
.LVL3301:
.L1997:
	movq	%rax, %rbx
.LVL3302:
	jmp	.L1970
.LVL3303:
.L1968:
	.loc 1 369 0 discriminator 2
	leaq	-352(%rbp), %rdi
.LVL3304:
	call	_ZN2cv3MatD1Ev
.LVL3305:
	jmp	.L1969
.LVL3306:
.L1970:
	.loc 1 370 0 discriminator 2
	leaq	-256(%rbp), %rdi
.LVL3307:
	call	_ZN2cv3MatD1Ev
.LVL3308:
	jmp	.L1969
	.cfi_endproc
.LFE11600:
	.section	.gcc_except_table
.LLSDA11600:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE11600-.LLSDACSB11600
.LLSDACSB11600:
	.uleb128 .LEHB66-.LFB11600
	.uleb128 .LEHE66-.LEHB66
	.uleb128 .L1995-.LFB11600
	.uleb128 0
	.uleb128 .LEHB67-.LFB11600
	.uleb128 .LEHE67-.LEHB67
	.uleb128 .L1996-.LFB11600
	.uleb128 0
	.uleb128 .LEHB68-.LFB11600
	.uleb128 .LEHE68-.LEHB68
	.uleb128 .L1995-.LFB11600
	.uleb128 0
	.uleb128 .LEHB69-.LFB11600
	.uleb128 .LEHE69-.LEHB69
	.uleb128 .L1997-.LFB11600
	.uleb128 0
	.uleb128 .LEHB70-.LFB11600
	.uleb128 .LEHE70-.LEHB70
	.uleb128 .L1995-.LFB11600
	.uleb128 0
	.uleb128 .LEHB71-.LFB11600
	.uleb128 .LEHE71-.LEHB71
	.uleb128 .L1998-.LFB11600
	.uleb128 0
	.uleb128 .LEHB72-.LFB11600
	.uleb128 .LEHE72-.LEHB72
	.uleb128 .L1996-.LFB11600
	.uleb128 0
	.uleb128 .LEHB73-.LFB11600
	.uleb128 .LEHE73-.LEHB73
	.uleb128 .L1997-.LFB11600
	.uleb128 0
	.uleb128 .LEHB74-.LFB11600
	.uleb128 .LEHE74-.LEHB74
	.uleb128 .L1998-.LFB11600
	.uleb128 0
	.uleb128 .LEHB75-.LFB11600
	.uleb128 .LEHE75-.LEHB75
	.uleb128 0
	.uleb128 0
.LLSDACSE11600:
	.section	.text._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_,comdat
	.size	_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_, .-_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_
	.section	.text.unlikely._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_,comdat
.LCOLDE78:
	.section	.text._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_,comdat
.LHOTE78:
	.section	.text.unlikely._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_,comdat
	.align 2
.LCOLDB79:
	.section	.text._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_,comdat
.LHOTB79:
	.align 2
	.p2align 4,,15
	.weak	_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_
	.type	_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_, @function
_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_:
.LFB11601:
	.loc 1 354 0
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA11601
.LVL3309:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdx, %rax
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$776, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	.loc 1 354 0
	movq	%rdx, -808(%rbp)
	.loc 1 359 0
	movl	(%rsi), %edx
.LVL3310:
	.loc 1 354 0
	movq	%fs:40, %rbx
	movq	%rbx, -56(%rbp)
	xorl	%ebx, %ebx
	.loc 1 356 0
	movl	8(%rsi), %ebx
	.loc 1 354 0
	movq	%rdi, -800(%rbp)
	.loc 1 359 0
	andl	$4095, %edx
	.loc 1 357 0
	movl	12(%rsi), %r14d
	.loc 1 356 0
	movl	%ebx, -788(%rbp)
.LVL3311:
	.loc 1 359 0
	movq	%rax, %rbx
.LVL3312:
	movl	(%rax), %eax
.LVL3313:
	movl	%eax, -792(%rbp)
	andl	$4095, %eax
	cmpl	%eax, %edx
	jne	.L2071
.LBB13753:
.LBB13754:
	.loc 2 713 0 discriminator 2
	movq	64(%rbx), %rax
.LBE13754:
.LBE13753:
.LBB13755:
.LBB13756:
	movq	64(%rsi), %rdx
	movq	%rsi, %r13
.LVL3314:
.LBE13756:
.LBE13755:
.LBB13757:
.LBB13758:
	.loc 13 1893 0 discriminator 2
	movl	(%rax), %ebx
.LVL3315:
	cmpl	%ebx, (%rdx)
	jne	.L2071
	movl	4(%rax), %eax
	cmpl	%eax, 4(%rdx)
	je	.L2072
.LVL3316:
.L2071:
.LBE13758:
.LBE13757:
	.loc 1 359 0 discriminator 7
	movl	$_ZZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_E19__PRETTY_FUNCTION__, %ecx
	movl	$359, %edx
	movl	$.LC45, %esi
.LVL3317:
	movl	$.LC76, %edi
.LVL3318:
	call	__assert_fail
.LVL3319:
	.p2align 4,,10
	.p2align 3
.L2072:
.LBB13759:
.LBB13760:
.LBB13761:
	.loc 2 709 0
	leaq	-640(%rbp), %rbx
.LVL3320:
.LBE13761:
.LBE13760:
.LBE13759:
.LBB13782:
.LBB13783:
.LBB13784:
	leaq	-544(%rbp), %r12
.LBE13784:
.LBE13783:
.LBE13782:
.LBB13801:
.LBB13764:
.LBB13765:
	.loc 2 62 0
	movq	$0, -592(%rbp)
.LBE13765:
.LBE13764:
.LBB13770:
.LBB13771:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE13771:
.LBE13770:
.LBB13774:
.LBB13766:
	.loc 2 62 0
	movq	$0, -600(%rbp)
.LBE13766:
.LBE13774:
.LBB13775:
.LBB13762:
	.loc 2 709 0
	leaq	8(%rbx), %rax
.LBE13762:
.LBE13775:
.LBB13776:
.LBB13767:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm5
	.loc 2 62 0
	movq	$0, -608(%rbp)
	movq	$0, -624(%rbp)
.LBE13767:
.LBE13776:
.LBE13801:
.LBB13802:
.LBB13803:
.LBB13804:
	.loc 2 353 0
	leaq	-688(%rbp), %rdx
.LBE13804:
.LBE13803:
.LBE13802:
.LBB13819:
.LBB13777:
.LBB13763:
	.loc 2 709 0
	movq	%rax, -576(%rbp)
.LVL3321:
.LBE13763:
.LBE13777:
.LBB13778:
.LBB13772:
	.loc 2 738 0
	leaq	80(%rbx), %rax
.LBE13772:
.LBE13778:
.LBB13779:
.LBB13768:
	.loc 2 63 0
	movq	$0, -616(%rbp)
.LBE13768:
.LBE13779:
.LBB13780:
.LBB13773:
	.loc 2 738 0
	movaps	%xmm0, -560(%rbp)
.LVL3322:
	movq	%rax, -568(%rbp)
.LBE13773:
.LBE13780:
.LBE13819:
.LBB13820:
.LBB13787:
.LBB13785:
	.loc 2 709 0
	leaq	8(%r12), %rax
.LBE13785:
.LBE13787:
.LBE13820:
.LBB13821:
.LBB13781:
.LBB13769:
	.loc 2 64 0
	movq	$0, -584(%rbp)
.LVL3323:
	.loc 2 60 0
	movaps	%xmm5, -640(%rbp)
.LBE13769:
.LBE13781:
.LBE13821:
.LBB13822:
.LBB13808:
.LBB13805:
	.loc 2 353 0
	leaq	-352(%rbp), %rdi
.LVL3324:
	movl	$21, %ecx
	movl	$2, %esi
.LVL3325:
.LBE13805:
.LBE13808:
.LBE13822:
.LBB13823:
.LBB13788:
.LBB13789:
	.loc 2 738 0
	movaps	%xmm0, -464(%rbp)
.LBE13789:
.LBE13788:
.LBB13792:
.LBB13793:
	.loc 2 60 0
	movaps	%xmm5, -544(%rbp)
.LBE13793:
.LBE13792:
.LBE13823:
.LBB13824:
.LBB13825:
.LBB13826:
	.loc 2 738 0
	movaps	%xmm0, -368(%rbp)
.LBE13826:
.LBE13825:
.LBB13830:
.LBB13831:
	.loc 2 60 0
	movaps	%xmm5, -448(%rbp)
.LBE13831:
.LBE13830:
.LBE13824:
.LBB13847:
.LBB13796:
.LBB13786:
	.loc 2 709 0
	movq	%rax, -480(%rbp)
.LVL3326:
.LBE13786:
.LBE13796:
.LBB13797:
.LBB13790:
	.loc 2 738 0
	leaq	80(%r12), %rax
.LBE13790:
.LBE13797:
.LBB13798:
.LBB13794:
	.loc 2 62 0
	movq	$0, -496(%rbp)
	movq	$0, -504(%rbp)
	movq	$0, -512(%rbp)
.LBE13794:
.LBE13798:
.LBB13799:
.LBB13791:
	.loc 2 738 0
	movq	%rax, -472(%rbp)
.LBE13791:
.LBE13799:
.LBE13847:
.LBB13848:
.LBB13835:
.LBB13836:
	.loc 2 709 0
	leaq	-448(%rbp), %rax
.LBE13836:
.LBE13835:
.LBE13848:
.LBB13849:
.LBB13800:
.LBB13795:
	.loc 2 62 0
	movq	$0, -528(%rbp)
	.loc 2 63 0
	movq	$0, -520(%rbp)
	.loc 2 64 0
	movq	$0, -488(%rbp)
.LVL3327:
.LBE13795:
.LBE13800:
.LBE13849:
.LBB13850:
.LBB13839:
.LBB13837:
	.loc 2 709 0
	addq	$8, %rax
.LVL3328:
.LBE13837:
.LBE13839:
.LBB13840:
.LBB13832:
	.loc 2 62 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
.LBE13832:
.LBE13840:
.LBB13841:
.LBB13838:
	.loc 2 709 0
	movq	%rax, -384(%rbp)
.LVL3329:
.LBE13838:
.LBE13841:
.LBB13842:
.LBB13827:
	.loc 2 738 0
	leaq	-448(%rbp), %rax
.LVL3330:
.LBE13827:
.LBE13842:
.LBB13843:
.LBB13833:
	.loc 2 62 0
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
	.loc 2 63 0
	movq	$0, -424(%rbp)
.LBE13833:
.LBE13843:
.LBB13844:
.LBB13828:
	.loc 2 738 0
	addq	$80, %rax
.LVL3331:
.LBE13828:
.LBE13844:
.LBB13845:
.LBB13834:
	.loc 2 64 0
	movq	$0, -392(%rbp)
.LVL3332:
.LBE13834:
.LBE13845:
.LBB13846:
.LBB13829:
	.loc 2 738 0
	movq	%rax, -376(%rbp)
.LBE13829:
.LBE13846:
.LBE13850:
.LBB13851:
.LBB13809:
.LBB13810:
	.loc 2 709 0
	leaq	-352(%rbp), %rax
	addq	$8, %rax
	movq	%rax, -288(%rbp)
.LVL3333:
.LBE13810:
.LBE13809:
.LBB13811:
.LBB13812:
	.loc 2 738 0
	leaq	-352(%rbp), %rax
	addq	$80, %rax
	movq	%rax, -280(%rbp)
.LBE13812:
.LBE13811:
.LBB13814:
.LBB13806:
	.loc 2 352 0
	movl	-788(%rbp), %eax
.LBE13806:
.LBE13814:
.LBB13815:
.LBB13813:
	.loc 2 738 0
	movaps	%xmm0, -272(%rbp)
.LVL3334:
.LBE13813:
.LBE13815:
.LBB13816:
.LBB13817:
	.loc 2 62 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
	.loc 2 60 0
	movaps	%xmm5, -352(%rbp)
	.loc 2 62 0
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
	.loc 2 63 0
	movq	$0, -328(%rbp)
	.loc 2 64 0
	movq	$0, -296(%rbp)
.LVL3335:
.LBE13817:
.LBE13816:
.LBB13818:
.LBB13807:
	.loc 2 352 0
	movl	%eax, -688(%rbp)
	movl	%r14d, -684(%rbp)
.LEHB76:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL3336:
.LEHE76:
.LBE13807:
.LBE13818:
.LBE13851:
.LBB13852:
.LBB13853:
	.loc 2 285 0
	movq	-328(%rbp), %rax
	leal	8(%r14), %r15d
.LVL3337:
	testq	%rax, %rax
	je	.L2074
	.loc 2 286 0
	lock addl	$1, (%rax)
.L2074:
.LVL3338:
.LBB13854:
.LBB13855:
	.loc 2 366 0
	movq	-616(%rbp), %rax
	testq	%rax, %rax
	je	.L2151
	lock subl	$1, (%rax)
	je	.L2238
.L2151:
.LBB13856:
	.loc 2 369 0
	movl	-636(%rbp), %ecx
.LBE13856:
	.loc 2 368 0
	movq	$0, -592(%rbp)
	movq	$0, -600(%rbp)
	movq	$0, -608(%rbp)
	movq	$0, -624(%rbp)
.LVL3339:
.LBB13857:
	.loc 2 369 0
	testl	%ecx, %ecx
	jle	.L2239
	movq	-576(%rbp), %rdx
	xorl	%eax, %eax
.LVL3340:
	.p2align 4,,10
	.p2align 3
.L2077:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-636(%rbp), %ecx
	addl	$1, %eax
.LVL3341:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L2077
.LBE13857:
.LBE13855:
.LBE13854:
	.loc 2 288 0
	movl	-352(%rbp), %eax
.LVL3342:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB13861:
.LBB13858:
	.loc 2 371 0
	movq	$0, -616(%rbp)
.LVL3343:
.LBE13858:
.LBE13861:
	.loc 2 288 0
	movl	%eax, -640(%rbp)
	.loc 2 289 0
	jg	.L2078
.L2166:
	movl	-348(%rbp), %eax
	cmpl	$2, %eax
	jle	.L2240
.L2078:
	.loc 2 298 0
	leaq	-352(%rbp), %rsi
.LVL3344:
	movq	%rbx, %rdi
.LEHB77:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL3345:
.LEHE77:
.L2079:
	.loc 2 299 0
	movdqa	-336(%rbp), %xmm0
	.loc 2 303 0
	movq	-328(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -624(%rbp)
.LBE13853:
.LBE13852:
.LBB13868:
.LBB13869:
.LBB13870:
.LBB13871:
	.loc 2 366 0
	testq	%rax, %rax
.LBE13871:
.LBE13870:
.LBE13869:
.LBE13868:
.LBB13879:
.LBB13864:
	.loc 2 299 0
	movdqa	-320(%rbp), %xmm0
	movaps	%xmm0, -608(%rbp)
	movdqa	-304(%rbp), %xmm0
	movaps	%xmm0, -592(%rbp)
.LVL3346:
.LBE13864:
.LBE13879:
.LBB13880:
.LBB13878:
.LBB13876:
.LBB13874:
	.loc 2 366 0
	je	.L2154
	lock subl	$1, (%rax)
	jne	.L2154
	.loc 2 367 0
	leaq	-352(%rbp), %rdi
.LVL3347:
	call	_ZN2cv3Mat10deallocateEv
.LVL3348:
.L2154:
.LBB13872:
	.loc 2 369 0
	movl	-348(%rbp), %r11d
.LBE13872:
	.loc 2 368 0
	movq	$0, -304(%rbp)
	movq	$0, -312(%rbp)
	movq	$0, -320(%rbp)
	movq	$0, -336(%rbp)
.LVL3349:
.LBB13873:
	.loc 2 369 0
	testl	%r11d, %r11d
	jle	.L2085
	movq	-288(%rbp), %rdx
	xorl	%eax, %eax
.LVL3350:
	.p2align 4,,10
	.p2align 3
.L2086:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3351:
	addq	$4, %rdx
	cmpl	%eax, -348(%rbp)
	jg	.L2086
.LVL3352:
.L2085:
.LBE13873:
.LBE13874:
.LBE13876:
	.loc 2 277 0
	movq	-280(%rbp), %rdi
	leaq	-352(%rbp), %rax
.LVL3353:
.LBB13877:
.LBB13875:
	.loc 2 371 0
	movq	$0, -328(%rbp)
.LVL3354:
.LBE13875:
.LBE13877:
	.loc 2 277 0
	addq	$80, %rax
.LVL3355:
	cmpq	%rax, %rdi
	je	.L2084
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3356:
.L2084:
.LBE13878:
.LBE13880:
.LBB13881:
.LBB13882:
.LBB13883:
	.loc 2 709 0
	leaq	-256(%rbp), %rax
.LVL3357:
.LBE13883:
.LBE13882:
.LBB13886:
.LBB13887:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE13887:
.LBE13886:
.LBB13891:
.LBB13892:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm3
.LBE13892:
.LBE13891:
.LBB13895:
.LBB13896:
	.loc 2 353 0
	leaq	-672(%rbp), %rdx
.LBE13896:
.LBE13895:
.LBB13901:
.LBB13884:
	.loc 2 709 0
	addq	$8, %rax
.LVL3358:
.LBE13884:
.LBE13901:
.LBB13902:
.LBB13897:
	.loc 2 353 0
	leaq	-256(%rbp), %rdi
.LVL3359:
	movl	$21, %ecx
.LBE13897:
.LBE13902:
.LBB13903:
.LBB13885:
	.loc 2 709 0
	movq	%rax, -192(%rbp)
.LVL3360:
.LBE13885:
.LBE13903:
.LBB13904:
.LBB13888:
	.loc 2 738 0
	leaq	-256(%rbp), %rax
.LBE13888:
.LBE13904:
.LBB13905:
.LBB13898:
	.loc 2 353 0
	movl	$2, %esi
.LBE13898:
.LBE13905:
.LBB13906:
.LBB13889:
	.loc 2 738 0
	movaps	%xmm0, -176(%rbp)
.LVL3361:
	addq	$80, %rax
.LBE13889:
.LBE13906:
.LBB13907:
.LBB13893:
	.loc 2 62 0
	movq	$0, -208(%rbp)
	movq	$0, -216(%rbp)
	.loc 2 60 0
	movaps	%xmm3, -256(%rbp)
.LBE13893:
.LBE13907:
.LBB13908:
.LBB13890:
	.loc 2 738 0
	movq	%rax, -184(%rbp)
.LBE13890:
.LBE13908:
.LBB13909:
.LBB13899:
	.loc 2 352 0
	movl	-788(%rbp), %eax
.LBE13899:
.LBE13909:
.LBB13910:
.LBB13894:
	.loc 2 62 0
	movq	$0, -224(%rbp)
	movq	$0, -240(%rbp)
	.loc 2 63 0
	movq	$0, -232(%rbp)
	.loc 2 64 0
	movq	$0, -200(%rbp)
.LVL3362:
.LBE13894:
.LBE13910:
.LBB13911:
.LBB13900:
	.loc 2 352 0
	movl	%eax, -672(%rbp)
	movl	%r14d, -668(%rbp)
.LEHB78:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL3363:
.LEHE78:
.LBE13900:
.LBE13911:
.LBE13881:
.LBB13912:
.LBB13913:
	.loc 2 285 0
	movq	-232(%rbp), %rax
	testq	%rax, %rax
	je	.L2087
	.loc 2 286 0
	lock addl	$1, (%rax)
.L2087:
.LVL3364:
.LBB13914:
.LBB13915:
	.loc 2 366 0
	movq	-520(%rbp), %rax
	testq	%rax, %rax
	je	.L2155
	lock subl	$1, (%rax)
	je	.L2241
.L2155:
.LBB13916:
	.loc 2 369 0
	movl	-540(%rbp), %edx
.LBE13916:
	.loc 2 368 0
	movq	$0, -496(%rbp)
	movq	$0, -504(%rbp)
	movq	$0, -512(%rbp)
	movq	$0, -528(%rbp)
.LVL3365:
.LBB13917:
	.loc 2 369 0
	testl	%edx, %edx
	jle	.L2242
	movq	-480(%rbp), %rdx
	xorl	%eax, %eax
.LVL3366:
	.p2align 4,,10
	.p2align 3
.L2090:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-540(%rbp), %ecx
	addl	$1, %eax
.LVL3367:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L2090
.LBE13917:
.LBE13915:
.LBE13914:
	.loc 2 288 0
	movl	-256(%rbp), %eax
.LVL3368:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB13921:
.LBB13918:
	.loc 2 371 0
	movq	$0, -520(%rbp)
.LVL3369:
.LBE13918:
.LBE13921:
	.loc 2 288 0
	movl	%eax, -544(%rbp)
	.loc 2 289 0
	jg	.L2091
.L2167:
	movl	-252(%rbp), %eax
	cmpl	$2, %eax
	jle	.L2243
.L2091:
	.loc 2 298 0
	leaq	-256(%rbp), %rsi
.LVL3370:
	movq	%r12, %rdi
.LEHB79:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL3371:
.LEHE79:
.L2092:
	.loc 2 299 0
	movdqa	-240(%rbp), %xmm0
	.loc 2 303 0
	movq	-232(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -528(%rbp)
.LBE13913:
.LBE13912:
.LBB13928:
.LBB13929:
.LBB13930:
.LBB13931:
	.loc 2 366 0
	testq	%rax, %rax
.LBE13931:
.LBE13930:
.LBE13929:
.LBE13928:
.LBB13939:
.LBB13924:
	.loc 2 299 0
	movdqa	-224(%rbp), %xmm0
	movaps	%xmm0, -512(%rbp)
	movdqa	-208(%rbp), %xmm0
	movaps	%xmm0, -496(%rbp)
.LVL3372:
.LBE13924:
.LBE13939:
.LBB13940:
.LBB13938:
.LBB13936:
.LBB13934:
	.loc 2 366 0
	je	.L2158
	lock subl	$1, (%rax)
	jne	.L2158
	.loc 2 367 0
	leaq	-256(%rbp), %rdi
.LVL3373:
	call	_ZN2cv3Mat10deallocateEv
.LVL3374:
.L2158:
.LBB13932:
	.loc 2 369 0
	movl	-252(%rbp), %r10d
.LBE13932:
	.loc 2 368 0
	movq	$0, -208(%rbp)
	movq	$0, -216(%rbp)
	movq	$0, -224(%rbp)
	movq	$0, -240(%rbp)
.LVL3375:
.LBB13933:
	.loc 2 369 0
	testl	%r10d, %r10d
	jle	.L2098
	movq	-192(%rbp), %rdx
	xorl	%eax, %eax
.LVL3376:
	.p2align 4,,10
	.p2align 3
.L2099:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3377:
	addq	$4, %rdx
	cmpl	%eax, -252(%rbp)
	jg	.L2099
.LVL3378:
.L2098:
.LBE13933:
.LBE13934:
.LBE13936:
	.loc 2 277 0
	movq	-184(%rbp), %rdi
	leaq	-256(%rbp), %rax
.LVL3379:
.LBB13937:
.LBB13935:
	.loc 2 371 0
	movq	$0, -232(%rbp)
.LVL3380:
.LBE13935:
.LBE13937:
	.loc 2 277 0
	addq	$80, %rax
.LVL3381:
	cmpq	%rax, %rdi
	je	.L2097
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3382:
.L2097:
.LBE13938:
.LBE13940:
.LBB13941:
.LBB13942:
.LBB13943:
	.loc 2 709 0
	leaq	-160(%rbp), %rax
.LVL3383:
.LBE13943:
.LBE13942:
.LBB13946:
.LBB13947:
	.loc 2 738 0
	pxor	%xmm0, %xmm0
.LBE13947:
.LBE13946:
.LBB13951:
.LBB13952:
	.loc 2 60 0
	movdqa	.LC47(%rip), %xmm4
.LBE13952:
.LBE13951:
.LBB13955:
.LBB13956:
	.loc 2 353 0
	leaq	-656(%rbp), %rdx
.LBE13956:
.LBE13955:
.LBB13960:
.LBB13944:
	.loc 2 709 0
	addq	$8, %rax
.LVL3384:
.LBE13944:
.LBE13960:
.LBB13961:
.LBB13957:
	.loc 2 353 0
	leaq	-160(%rbp), %rdi
.LVL3385:
	movl	$21, %ecx
.LBE13957:
.LBE13961:
.LBB13962:
.LBB13945:
	.loc 2 709 0
	movq	%rax, -96(%rbp)
.LVL3386:
.LBE13945:
.LBE13962:
.LBB13963:
.LBB13948:
	.loc 2 738 0
	leaq	-160(%rbp), %rax
.LBE13948:
.LBE13963:
.LBB13964:
.LBB13958:
	.loc 2 353 0
	movl	$2, %esi
.LBE13958:
.LBE13964:
.LBB13965:
.LBB13949:
	.loc 2 738 0
	movaps	%xmm0, -80(%rbp)
.LVL3387:
	addq	$80, %rax
.LBE13949:
.LBE13965:
.LBB13966:
.LBB13953:
	.loc 2 62 0
	movq	$0, -112(%rbp)
	movq	$0, -120(%rbp)
	.loc 2 60 0
	movaps	%xmm4, -160(%rbp)
.LBE13953:
.LBE13966:
.LBB13967:
.LBB13950:
	.loc 2 738 0
	movq	%rax, -88(%rbp)
.LBE13950:
.LBE13967:
.LBB13968:
.LBB13954:
	.loc 2 62 0
	movq	$0, -128(%rbp)
	movq	$0, -144(%rbp)
	.loc 2 63 0
	movq	$0, -136(%rbp)
	.loc 2 64 0
	movq	$0, -104(%rbp)
.LVL3388:
.LBE13954:
.LBE13968:
.LBB13969:
.LBB13959:
	.loc 2 352 0
	movl	$4, -656(%rbp)
	movl	%r15d, -652(%rbp)
.LEHB80:
	.loc 2 353 0
	call	_ZN2cv3Mat6createEiPKii
.LVL3389:
.LEHE80:
.LBE13959:
.LBE13969:
.LBE13941:
.LBB13970:
.LBB13971:
	.loc 2 285 0
	movq	-136(%rbp), %rax
	testq	%rax, %rax
	je	.L2100
	.loc 2 286 0
	lock addl	$1, (%rax)
.L2100:
.LVL3390:
.LBB13972:
.LBB13973:
	.loc 2 366 0
	movq	-424(%rbp), %rax
	testq	%rax, %rax
	je	.L2159
	lock subl	$1, (%rax)
	je	.L2244
.L2159:
.LBB13974:
	.loc 2 369 0
	movl	-444(%rbp), %eax
.LBE13974:
	.loc 2 368 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
.LVL3391:
.LBB13975:
	.loc 2 369 0
	testl	%eax, %eax
	jle	.L2245
	movq	-384(%rbp), %rdx
	xorl	%eax, %eax
.LVL3392:
	.p2align 4,,10
	.p2align 3
.L2103:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	movl	-444(%rbp), %ecx
	addl	$1, %eax
.LVL3393:
	addq	$4, %rdx
	cmpl	%eax, %ecx
	jg	.L2103
.LBE13975:
.LBE13973:
.LBE13972:
	.loc 2 288 0
	movl	-160(%rbp), %eax
.LVL3394:
	.loc 2 289 0
	cmpl	$2, %ecx
.LBB13979:
.LBB13976:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL3395:
.LBE13976:
.LBE13979:
	.loc 2 288 0
	movl	%eax, -448(%rbp)
	.loc 2 289 0
	jg	.L2104
.L2168:
	movl	-156(%rbp), %eax
	cmpl	$2, %eax
	jle	.L2246
.L2104:
	.loc 2 298 0
	leaq	-160(%rbp), %rsi
.LVL3396:
	leaq	-448(%rbp), %rdi
.LVL3397:
.LEHB81:
	call	_ZN2cv3Mat8copySizeERKS0_
.LVL3398:
.LEHE81:
.L2105:
	.loc 2 299 0
	movdqa	-144(%rbp), %xmm0
	.loc 2 303 0
	movq	-136(%rbp), %rax
	.loc 2 299 0
	movaps	%xmm0, -432(%rbp)
.LBE13971:
.LBE13970:
.LBB13986:
.LBB13987:
.LBB13988:
.LBB13989:
	.loc 2 366 0
	testq	%rax, %rax
.LBE13989:
.LBE13988:
.LBE13987:
.LBE13986:
.LBB13997:
.LBB13982:
	.loc 2 299 0
	movdqa	-128(%rbp), %xmm0
	movaps	%xmm0, -416(%rbp)
	movdqa	-112(%rbp), %xmm0
	movaps	%xmm0, -400(%rbp)
.LVL3399:
.LBE13982:
.LBE13997:
.LBB13998:
.LBB13996:
.LBB13994:
.LBB13992:
	.loc 2 366 0
	je	.L2162
	lock subl	$1, (%rax)
	jne	.L2162
	.loc 2 367 0
	leaq	-160(%rbp), %rdi
.LVL3400:
	call	_ZN2cv3Mat10deallocateEv
.LVL3401:
.L2162:
.LBB13990:
	.loc 2 369 0
	movl	-156(%rbp), %r9d
.LBE13990:
	.loc 2 368 0
	movq	$0, -112(%rbp)
	movq	$0, -120(%rbp)
	movq	$0, -128(%rbp)
	movq	$0, -144(%rbp)
.LVL3402:
.LBB13991:
	.loc 2 369 0
	testl	%r9d, %r9d
	jle	.L2111
	movq	-96(%rbp), %rdx
	xorl	%eax, %eax
.LVL3403:
	.p2align 4,,10
	.p2align 3
.L2112:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3404:
	addq	$4, %rdx
	cmpl	%eax, -156(%rbp)
	jg	.L2112
.LVL3405:
.L2111:
.LBE13991:
.LBE13992:
.LBE13994:
	.loc 2 277 0
	movq	-88(%rbp), %rdi
	leaq	-160(%rbp), %rax
.LVL3406:
.LBB13995:
.LBB13993:
	.loc 2 371 0
	movq	$0, -136(%rbp)
.LVL3407:
.LBE13993:
.LBE13995:
	.loc 2 277 0
	addq	$80, %rax
.LVL3408:
	cmpq	%rax, %rdi
	je	.L2110
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3409:
.L2110:
.LBE13996:
.LBE13998:
	.loc 1 379 0
	leal	(%r14,%r14,2), %r15d
.LVL3410:
	movq	16(%r13), %rdx
	.loc 1 384 0
	movslq	%r15d, %rcx
	.loc 1 383 0
	movl	%r15d, %r14d
.LVL3411:
	.loc 1 384 0
	leaq	0(,%rcx,4), %rsi
	.loc 1 383 0
	sarl	$6, %r14d
	sall	$4, %r14d
	.loc 1 384 0
	leaq	18(%rsi), %rax
	.loc 1 383 0
	movl	%r14d, -792(%rbp)
.LVL3412:
	.loc 1 384 0
	andq	$-16, %rax
	subq	%rax, %rsp
	leaq	3(%rsp), %rdi
	shrq	$2, %rdi
.LBB13999:
	.loc 1 386 0
	testl	%r15d, %r15d
.LBE13999:
	.loc 1 384 0
	leaq	0(,%rdi,4), %r14
.LVL3413:
.LBB14000:
	.loc 1 386 0
	jle	.L2123
	addq	%r14, %rsi
	cmpq	%rsi, %rdx
	setnb	%sil
	addq	%rdx, %rcx
	cmpq	%rcx, %r14
	setnb	%cl
	orb	%cl, %sil
	je	.L2116
	cmpl	$20, %r15d
	jbe	.L2116
	movq	%r14, %rsi
	andl	$15, %esi
	shrq	$2, %rsi
	negq	%rsi
	andl	$3, %esi
	cmpl	%r15d, %esi
	cmova	%r15d, %esi
	xorl	%ecx, %ecx
	testl	%esi, %esi
	je	.L2117
	.loc 1 387 0
	movzbl	(%rdx), %ecx
	pxor	%xmm0, %xmm0
	cmpl	$1, %esi
	cvtsi2ss	%ecx, %xmm0
	.loc 1 386 0
	movl	$1, %ecx
	.loc 1 387 0
	movss	%xmm0, 0(,%rdi,4)
.LVL3414:
	je	.L2117
	movzbl	1(%rdx), %ecx
	pxor	%xmm0, %xmm0
	cmpl	$2, %esi
	cvtsi2ss	%ecx, %xmm0
	.loc 1 386 0
	movl	$2, %ecx
	.loc 1 387 0
	movss	%xmm0, 4(,%rdi,4)
.LVL3415:
	je	.L2117
	movzbl	2(%rdx), %ecx
	pxor	%xmm0, %xmm0
	cvtsi2ss	%ecx, %xmm0
	.loc 1 386 0
	movl	$3, %ecx
	.loc 1 387 0
	movss	%xmm0, 8(,%rdi,4)
.LVL3416:
.L2117:
	movl	%r15d, %r9d
	movl	%esi, %edi
	.loc 1 386 0
	xorl	%r10d, %r10d
	subl	%esi, %r9d
	leaq	(%r14,%rdi,4), %r8
	addq	%rdx, %rdi
	leal	-16(%r9), %esi
	shrl	$4, %esi
	addl	$1, %esi
	movl	%esi, %r11d
	sall	$4, %r11d
.L2120:
	.loc 1 387 0 discriminator 2
	movdqu	(%rdi), %xmm0
	addl	$1, %r10d
	addq	$64, %r8
	addq	$16, %rdi
	pmovzxbw	%xmm0, %xmm1
	psrldq	$8, %xmm0
	pmovzxbw	%xmm0, %xmm0
	pmovsxwd	%xmm1, %xmm2
	psrldq	$8, %xmm1
	pmovsxwd	%xmm1, %xmm1
	cvtdq2ps	%xmm2, %xmm2
	movaps	%xmm2, -64(%r8)
	cvtdq2ps	%xmm1, %xmm1
	movaps	%xmm1, -48(%r8)
	pmovsxwd	%xmm0, %xmm1
	psrldq	$8, %xmm0
	pmovsxwd	%xmm0, %xmm0
	cvtdq2ps	%xmm1, %xmm1
	movaps	%xmm1, -32(%r8)
	cvtdq2ps	%xmm0, %xmm0
	movaps	%xmm0, -16(%r8)
	cmpl	%r10d, %esi
	ja	.L2120
	addl	%r11d, %ecx
	cmpl	%r9d, %r11d
	je	.L2123
.LVL3417:
	.loc 1 387 0 is_stmt 0
	movslq	%ecx, %rdi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rdi), %esi
	cvtsi2ss	%esi, %xmm0
	.loc 1 386 0 is_stmt 1
	leal	1(%rcx), %esi
.LVL3418:
	cmpl	%esi, %r15d
	.loc 1 387 0
	movss	%xmm0, (%r14,%rdi,4)
	.loc 1 386 0
	jle	.L2123
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%r14,%rsi,4)
	.loc 1 386 0
	leal	2(%rcx), %esi
.LVL3419:
	cmpl	%esi, %r15d
	jle	.L2123
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%r14,%rsi,4)
	.loc 1 386 0
	leal	3(%rcx), %esi
.LVL3420:
	cmpl	%esi, %r15d
	jle	.L2123
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%r14,%rsi,4)
	.loc 1 386 0
	leal	4(%rcx), %esi
.LVL3421:
	cmpl	%esi, %r15d
	jle	.L2123
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%r14,%rsi,4)
	.loc 1 386 0
	leal	5(%rcx), %esi
.LVL3422:
	cmpl	%esi, %r15d
	jle	.L2123
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%r14,%rsi,4)
	.loc 1 386 0
	leal	6(%rcx), %esi
.LVL3423:
	cmpl	%esi, %r15d
	jle	.L2123
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%r14,%rsi,4)
	.loc 1 386 0
	leal	7(%rcx), %esi
.LVL3424:
	cmpl	%esi, %r15d
	jle	.L2123
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%r14,%rsi,4)
	.loc 1 386 0
	leal	8(%rcx), %esi
.LVL3425:
	cmpl	%esi, %r15d
	jle	.L2123
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%r14,%rsi,4)
	.loc 1 386 0
	leal	9(%rcx), %esi
.LVL3426:
	cmpl	%esi, %r15d
	jle	.L2123
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%r14,%rsi,4)
	.loc 1 386 0
	leal	10(%rcx), %esi
.LVL3427:
	cmpl	%esi, %r15d
	jle	.L2123
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%r14,%rsi,4)
	.loc 1 386 0
	leal	11(%rcx), %esi
.LVL3428:
	cmpl	%esi, %r15d
	jle	.L2123
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%r14,%rsi,4)
	.loc 1 386 0
	leal	12(%rcx), %esi
.LVL3429:
	cmpl	%esi, %r15d
	jle	.L2123
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%r14,%rsi,4)
	.loc 1 386 0
	leal	13(%rcx), %esi
.LVL3430:
	cmpl	%esi, %r15d
	jle	.L2123
	.loc 1 387 0
	movslq	%esi, %rsi
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rsi), %edi
	.loc 1 386 0
	addl	$14, %ecx
	cmpl	%ecx, %r15d
	.loc 1 387 0
	cvtsi2ss	%edi, %xmm0
	movss	%xmm0, (%r14,%rsi,4)
.LVL3431:
	.loc 1 386 0
	jle	.L2123
	.loc 1 387 0
	movslq	%ecx, %rcx
	pxor	%xmm0, %xmm0
	movzbl	(%rdx,%rcx), %edx
	cvtsi2ss	%edx, %xmm0
	movss	%xmm0, (%r14,%rcx,4)
.L2123:
.LBE14000:
	.loc 1 390 0
	call	omp_get_wtime
.LVL3432:
.LBB14001:
	.loc 1 392 0
	movl	-792(%rbp), %eax
	pxor	%xmm0, %xmm0
	leaq	-784(%rbp), %rsi
	xorl	%ecx, %ecx
	movl	$4, %edx
	movl	$_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_._omp_fn.5, %edi
	movq	%r14, -760(%rbp)
	movq	%r13, -752(%rbp)
	movl	%eax, -704(%rbp)
	movq	-800(%rbp), %rax
	movaps	%xmm0, -784(%rbp)
	movq	%rbx, -736(%rbp)
	movq	%r12, -720(%rbp)
	movq	%rax, -768(%rbp)
	movq	-808(%rbp), %rax
	movl	%r15d, -708(%rbp)
	movq	%rax, -744(%rbp)
	leaq	-448(%rbp), %rax
	movq	%rax, -728(%rbp)
	movl	-788(%rbp), %eax
	movl	%eax, -712(%rbp)
	call	GOMP_parallel
.LVL3433:
.LBE14001:
	.loc 1 608 0
	call	omp_get_wtime
.LVL3434:
.LBB14002:
.LBB14003:
.LBB14004:
.LBB14005:
	.loc 2 366 0
	movq	-424(%rbp), %rax
	testq	%rax, %rax
	je	.L2163
	lock subl	$1, (%rax)
	jne	.L2163
	.loc 2 367 0
	leaq	-448(%rbp), %rdi
.LVL3435:
	call	_ZN2cv3Mat10deallocateEv
.LVL3436:
.L2163:
.LBB14006:
	.loc 2 369 0
	movl	-444(%rbp), %r8d
.LBE14006:
	.loc 2 368 0
	movq	$0, -400(%rbp)
	movq	$0, -408(%rbp)
	movq	$0, -416(%rbp)
	movq	$0, -432(%rbp)
.LVL3437:
.LBB14007:
	.loc 2 369 0
	testl	%r8d, %r8d
	jle	.L2130
	movq	-384(%rbp), %rdx
	xorl	%eax, %eax
.LVL3438:
	.p2align 4,,10
	.p2align 3
.L2131:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3439:
	addq	$4, %rdx
	cmpl	%eax, -444(%rbp)
	jg	.L2131
.LVL3440:
.L2130:
.LBE14007:
.LBE14005:
.LBE14004:
	.loc 2 277 0
	movq	-376(%rbp), %rdi
	leaq	-448(%rbp), %rax
.LVL3441:
.LBB14009:
.LBB14008:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL3442:
.LBE14008:
.LBE14009:
	.loc 2 277 0
	addq	$80, %rax
.LVL3443:
	cmpq	%rax, %rdi
	je	.L2129
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3444:
.L2129:
.LBE14003:
.LBE14002:
.LBB14010:
.LBB14011:
.LBB14012:
.LBB14013:
	.loc 2 366 0
	movq	-520(%rbp), %rax
	testq	%rax, %rax
	je	.L2164
	lock subl	$1, (%rax)
	jne	.L2164
	.loc 2 367 0
	movq	%r12, %rdi
	call	_ZN2cv3Mat10deallocateEv
.LVL3445:
.L2164:
.LBB14014:
	.loc 2 369 0
	movl	-540(%rbp), %edi
.LBE14014:
	.loc 2 368 0
	movq	$0, -496(%rbp)
	movq	$0, -504(%rbp)
	movq	$0, -512(%rbp)
	movq	$0, -528(%rbp)
.LVL3446:
.LBB14015:
	.loc 2 369 0
	testl	%edi, %edi
	jle	.L2137
	movq	-480(%rbp), %rdx
	xorl	%eax, %eax
.LVL3447:
	.p2align 4,,10
	.p2align 3
.L2138:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3448:
	addq	$4, %rdx
	cmpl	%eax, -540(%rbp)
	jg	.L2138
.LVL3449:
.L2137:
.LBE14015:
.LBE14013:
.LBE14012:
	.loc 2 277 0
	movq	-472(%rbp), %rdi
	addq	$80, %r12
.LVL3450:
.LBB14017:
.LBB14016:
	.loc 2 371 0
	movq	$0, -520(%rbp)
.LVL3451:
.LBE14016:
.LBE14017:
	.loc 2 277 0
	cmpq	%r12, %rdi
	je	.L2136
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3452:
.L2136:
.LBE14011:
.LBE14010:
.LBB14018:
.LBB14019:
.LBB14020:
.LBB14021:
	.loc 2 366 0
	movq	-616(%rbp), %rax
	testq	%rax, %rax
	je	.L2165
	lock subl	$1, (%rax)
	jne	.L2165
	.loc 2 367 0
	movq	%rbx, %rdi
	call	_ZN2cv3Mat10deallocateEv
.LVL3453:
.L2165:
.LBB14022:
	.loc 2 369 0
	movl	-636(%rbp), %esi
.LBE14022:
	.loc 2 368 0
	movq	$0, -592(%rbp)
	movq	$0, -600(%rbp)
	movq	$0, -608(%rbp)
	movq	$0, -624(%rbp)
.LVL3454:
.LBB14023:
	.loc 2 369 0
	testl	%esi, %esi
	jle	.L2144
	movq	-576(%rbp), %rdx
	xorl	%eax, %eax
.LVL3455:
	.p2align 4,,10
	.p2align 3
.L2145:
	.loc 2 370 0
	movl	$0, (%rdx)
	.loc 2 369 0
	addl	$1, %eax
.LVL3456:
	addq	$4, %rdx
	cmpl	%eax, -636(%rbp)
	jg	.L2145
.LVL3457:
.L2144:
.LBE14023:
.LBE14021:
.LBE14020:
	.loc 2 277 0
	movq	-568(%rbp), %rdi
	addq	$80, %rbx
.LVL3458:
.LBB14025:
.LBB14024:
	.loc 2 371 0
	movq	$0, -616(%rbp)
.LVL3459:
.LBE14024:
.LBE14025:
	.loc 2 277 0
	cmpq	%rbx, %rdi
	je	.L2070
	.loc 2 278 0
	call	_ZN2cv8fastFreeEPv
.LVL3460:
.L2070:
.LBE14019:
.LBE14018:
	.loc 1 609 0
	movq	-56(%rbp), %rax
	xorq	%fs:40, %rax
	jne	.L2247
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
.LVL3461:
	popq	%r14
.LVL3462:
	popq	%r15
.LVL3463:
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
.LVL3464:
	ret
.LVL3465:
	.p2align 4,,10
	.p2align 3
.L2240:
	.cfi_restore_state
.LBB14026:
.LBB13865:
	.loc 2 291 0
	movl	%eax, -636(%rbp)
	.loc 2 292 0
	movl	-344(%rbp), %eax
	movq	-280(%rbp), %rdx
	movl	%eax, -632(%rbp)
	.loc 2 293 0
	movl	-340(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -628(%rbp)
	movq	-568(%rbp), %rax
.LVL3466:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL3467:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L2079
.LVL3468:
	.p2align 4,,10
	.p2align 3
.L2243:
.LBE13865:
.LBE14026:
.LBB14027:
.LBB13925:
	.loc 2 291 0
	movl	%eax, -540(%rbp)
	.loc 2 292 0
	movl	-248(%rbp), %eax
	movq	-184(%rbp), %rdx
	movl	%eax, -536(%rbp)
	.loc 2 293 0
	movl	-244(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -532(%rbp)
	movq	-472(%rbp), %rax
.LVL3469:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL3470:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L2092
.LVL3471:
	.p2align 4,,10
	.p2align 3
.L2246:
.LBE13925:
.LBE14027:
.LBB14028:
.LBB13983:
	.loc 2 291 0
	movl	%eax, -444(%rbp)
	.loc 2 292 0
	movl	-152(%rbp), %eax
	movq	-88(%rbp), %rdx
	movl	%eax, -440(%rbp)
	.loc 2 293 0
	movl	-148(%rbp), %eax
	.loc 2 294 0
	movq	(%rdx), %rcx
	.loc 2 293 0
	movl	%eax, -436(%rbp)
	movq	-376(%rbp), %rax
.LVL3472:
	.loc 2 294 0
	movq	%rcx, (%rax)
.LVL3473:
	.loc 2 295 0
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	jmp	.L2105
.LVL3474:
	.p2align 4,,10
	.p2align 3
.L2238:
.LBE13983:
.LBE14028:
.LBB14029:
.LBB13866:
.LBB13862:
.LBB13859:
	.loc 2 367 0
	movq	%rbx, %rdi
.LEHB82:
	call	_ZN2cv3Mat10deallocateEv
.LVL3475:
.LEHE82:
	jmp	.L2151
.LVL3476:
	.p2align 4,,10
	.p2align 3
.L2241:
.LBE13859:
.LBE13862:
.LBE13866:
.LBE14029:
.LBB14030:
.LBB13926:
.LBB13922:
.LBB13919:
	movq	%r12, %rdi
.LEHB83:
	call	_ZN2cv3Mat10deallocateEv
.LVL3477:
.LEHE83:
	jmp	.L2155
.LVL3478:
	.p2align 4,,10
	.p2align 3
.L2244:
.LBE13919:
.LBE13922:
.LBE13926:
.LBE14030:
.LBB14031:
.LBB13984:
.LBB13980:
.LBB13977:
	leaq	-448(%rbp), %rdi
.LVL3479:
.LEHB84:
	call	_ZN2cv3Mat10deallocateEv
.LVL3480:
.LEHE84:
	jmp	.L2159
.LVL3481:
	.p2align 4,,10
	.p2align 3
.L2116:
.LBE13977:
.LBE13980:
.LBE13984:
.LBE14031:
.LBB14032:
	.loc 1 386 0
	xorl	%ecx, %ecx
.LVL3482:
	.p2align 4,,10
	.p2align 3
.L2125:
	.loc 1 387 0
	movzbl	(%rdx,%rcx), %eax
	pxor	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	movss	%xmm0, (%r14,%rcx,4)
.LVL3483:
	addq	$1, %rcx
.LVL3484:
	.loc 1 386 0
	cmpl	%ecx, %r15d
	jg	.L2125
	jmp	.L2123
.LVL3485:
.L2245:
.LBE14032:
.LBB14033:
.LBB13985:
	.loc 2 288 0
	movl	-160(%rbp), %eax
.LBB13981:
.LBB13978:
	.loc 2 371 0
	movq	$0, -424(%rbp)
.LVL3486:
.LBE13978:
.LBE13981:
	.loc 2 288 0
	movl	%eax, -448(%rbp)
	jmp	.L2168
.LVL3487:
.L2242:
.LBE13985:
.LBE14033:
.LBB14034:
.LBB13927:
	movl	-256(%rbp), %eax
.LBB13923:
.LBB13920:
	.loc 2 371 0
	movq	$0, -520(%rbp)
.LVL3488:
.LBE13920:
.LBE13923:
	.loc 2 288 0
	movl	%eax, -544(%rbp)
	jmp	.L2167
.LVL3489:
.L2239:
.LBE13927:
.LBE14034:
.LBB14035:
.LBB13867:
	movl	-352(%rbp), %eax
.LBB13863:
.LBB13860:
	.loc 2 371 0
	movq	$0, -616(%rbp)
.LVL3490:
.LBE13860:
.LBE13863:
	.loc 2 288 0
	movl	%eax, -640(%rbp)
	jmp	.L2166
.LVL3491:
.L2247:
.LBE13867:
.LBE14035:
	.loc 1 609 0
	call	__stack_chk_fail
.LVL3492:
.L2176:
	movq	%rax, %r13
.LVL3493:
	jmp	.L2149
.LVL3494:
.L2173:
	movq	%rax, %r13
.LVL3495:
	jmp	.L2147
.LVL3496:
.L2149:
	.loc 1 376 0 discriminator 2
	leaq	-160(%rbp), %rdi
.LVL3497:
	call	_ZN2cv3MatD1Ev
.LVL3498:
.L2147:
	.loc 1 362 0
	leaq	-448(%rbp), %rdi
	call	_ZN2cv3MatD1Ev
.LVL3499:
	.loc 1 361 0
	movq	%r12, %rdi
	call	_ZN2cv3MatD1Ev
.LVL3500:
	.loc 1 360 0
	movq	%rbx, %rdi
	call	_ZN2cv3MatD1Ev
.LVL3501:
	movq	%r13, %rdi
.LEHB85:
	call	_Unwind_Resume
.LVL3502:
.LEHE85:
.L2174:
	movq	%rax, %r13
.LVL3503:
	jmp	.L2146
.LVL3504:
.L2175:
	movq	%rax, %r13
.LVL3505:
	jmp	.L2148
.LVL3506:
.L2146:
	.loc 1 374 0 discriminator 2
	leaq	-352(%rbp), %rdi
.LVL3507:
	call	_ZN2cv3MatD1Ev
.LVL3508:
	jmp	.L2147
.LVL3509:
.L2148:
	.loc 1 375 0 discriminator 2
	leaq	-256(%rbp), %rdi
.LVL3510:
	call	_ZN2cv3MatD1Ev
.LVL3511:
	jmp	.L2147
	.cfi_endproc
.LFE11601:
	.section	.gcc_except_table
.LLSDA11601:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE11601-.LLSDACSB11601
.LLSDACSB11601:
	.uleb128 .LEHB76-.LFB11601
	.uleb128 .LEHE76-.LEHB76
	.uleb128 .L2173-.LFB11601
	.uleb128 0
	.uleb128 .LEHB77-.LFB11601
	.uleb128 .LEHE77-.LEHB77
	.uleb128 .L2174-.LFB11601
	.uleb128 0
	.uleb128 .LEHB78-.LFB11601
	.uleb128 .LEHE78-.LEHB78
	.uleb128 .L2173-.LFB11601
	.uleb128 0
	.uleb128 .LEHB79-.LFB11601
	.uleb128 .LEHE79-.LEHB79
	.uleb128 .L2175-.LFB11601
	.uleb128 0
	.uleb128 .LEHB80-.LFB11601
	.uleb128 .LEHE80-.LEHB80
	.uleb128 .L2173-.LFB11601
	.uleb128 0
	.uleb128 .LEHB81-.LFB11601
	.uleb128 .LEHE81-.LEHB81
	.uleb128 .L2176-.LFB11601
	.uleb128 0
	.uleb128 .LEHB82-.LFB11601
	.uleb128 .LEHE82-.LEHB82
	.uleb128 .L2174-.LFB11601
	.uleb128 0
	.uleb128 .LEHB83-.LFB11601
	.uleb128 .LEHE83-.LEHB83
	.uleb128 .L2175-.LFB11601
	.uleb128 0
	.uleb128 .LEHB84-.LFB11601
	.uleb128 .LEHE84-.LEHB84
	.uleb128 .L2176-.LFB11601
	.uleb128 0
	.uleb128 .LEHB85-.LFB11601
	.uleb128 .LEHE85-.LEHB85
	.uleb128 0
	.uleb128 0
.LLSDACSE11601:
	.section	.text._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_,comdat
	.size	_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_, .-_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_
	.section	.text.unlikely._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_,comdat
.LCOLDE79:
	.section	.text._ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_,"axG",@progbits,_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_,comdat
.LHOTE79:
	.section	.text.unlikely
	.align 2
.LCOLDB80:
	.text
.LHOTB80:
	.align 2
	.p2align 4,,15
	.globl	_ZN15fastGaussianIIR5applyERKN2cv3MatERS1_
	.type	_ZN15fastGaussianIIR5applyERKN2cv3MatERS1_, @function
_ZN15fastGaussianIIR5applyERKN2cv3MatERS1_:
.LFB11283:
	.loc 1 612 0
	.cfi_startproc
.LVL3512:
.LBB14038:
.LBB14039:
	.loc 2 402 0
	movl	(%rsi), %eax
	andl	$4088, %eax
	sarl	$3, %eax
	addl	$1, %eax
.LBE14039:
.LBE14038:
	.loc 1 613 0
	cmpl	$1, %eax
	je	.L2252
.LVL3513:
	.loc 1 616 0
	cmpl	$2, %eax
	je	.L2253
.LVL3514:
	.loc 1 619 0
	cmpl	$3, %eax
	je	.L2254
	rep ret
	.p2align 4,,10
	.p2align 3
.L2254:
	.loc 1 620 0
	jmp	_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_
.LVL3515:
	.p2align 4,,10
	.p2align 3
.L2252:
	.loc 1 614 0
	jmp	_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_
.LVL3516:
	.p2align 4,,10
	.p2align 3
.L2253:
	.loc 1 617 0
	jmp	_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_
.LVL3517:
	.cfi_endproc
.LFE11283:
	.size	_ZN15fastGaussianIIR5applyERKN2cv3MatERS1_, .-_ZN15fastGaussianIIR5applyERKN2cv3MatERS1_
	.section	.text.unlikely
.LCOLDE80:
	.text
.LHOTE80:
	.section	.rodata.str1.8
	.align 8
.LC81:
	.string	"std::abs(gblur->sigmaX - sigmaX) < EPSILON && std::abs(gblur->sigmaY - sigmaY) < EPSILON"
	.section	.text.unlikely
.LCOLDB82:
	.text
.LHOTB82:
	.p2align 4,,15
	.globl	_Z21constTimeGaussianBlurRKN2cv3MatERS0_ddPK15fastGaussianIIR
	.type	_Z21constTimeGaussianBlurRKN2cv3MatERS0_ddPK15fastGaussianIIR, @function
_Z21constTimeGaussianBlurRKN2cv3MatERS0_ddPK15fastGaussianIIR:
.LFB11287:
	.loc 1 710 0
	.cfi_startproc
.LVL3518:
.LBB14073:
.LBB14074:
	.loc 6 226 0
	movapd	%xmm1, %xmm3
.LBE14074:
.LBE14073:
	.loc 1 710 0
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
.LBB14076:
	.loc 1 712 0
	movsd	.LC67(%rip), %xmm2
.LBE14076:
.LBB14107:
.LBB14075:
	.loc 6 226 0
	maxsd	%xmm0, %xmm3
.LBE14075:
.LBE14107:
	.loc 1 710 0
	movq	%fs:40, %rax
	movq	%rax, 8(%rsp)
	xorl	%eax, %eax
.LVL3519:
.LBB14108:
	.loc 1 712 0
	ucomisd	%xmm3, %xmm2
	jnb	.L2258
	.loc 1 712 0 is_stmt 0 discriminator 1
	testq	%rdx, %rdx
	movq	%rdx, %rcx
	je	.L2258
	.loc 1 717 0 is_stmt 1
	movsd	(%rdx), %xmm2
	movsd	.LC41(%rip), %xmm4
	subsd	%xmm0, %xmm2
	movsd	.LC42(%rip), %xmm3
	andpd	%xmm4, %xmm2
	ucomisd	%xmm2, %xmm3
	jbe	.L2264
	.loc 1 717 0 is_stmt 0 discriminator 2
	movsd	8(%rdx), %xmm0
.LVL3520:
	subsd	%xmm1, %xmm0
	andpd	%xmm4, %xmm0
	ucomisd	%xmm0, %xmm3
	jbe	.L2264
.LVL3521:
.LBB14077:
.LBB14078:
.LBB14079:
.LBB14080:
	.loc 2 402 0 is_stmt 1
	movl	(%rdi), %eax
	andl	$4088, %eax
	sarl	$3, %eax
	addl	$1, %eax
.LBE14080:
.LBE14079:
	.loc 1 613 0
	cmpl	$1, %eax
	je	.L2274
.LVL3522:
	.loc 1 616 0
	cmpl	$2, %eax
	je	.L2275
.LVL3523:
	.loc 1 619 0
	cmpl	$3, %eax
	je	.L2276
.LVL3524:
.L2255:
.LBE14078:
.LBE14077:
.LBE14108:
	.loc 1 720 0
	movq	8(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L2277
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.LVL3525:
	.p2align 4,,10
	.p2align 3
.L2258:
	.cfi_restore_state
.LBB14109:
.LBB14083:
	.loc 1 713 0
	movapd	%xmm0, %xmm3
	mulsd	%xmm2, %xmm3
	.loc 1 714 0
	mulsd	%xmm1, %xmm2
	.loc 1 713 0
	cvttsd2si	%xmm3, %eax
	leal	1(%rax,%rax), %ecx
.LVL3526:
	.loc 1 714 0
	cvttsd2si	%xmm2, %eax
	leal	1(%rax,%rax), %edx
.LVL3527:
.LBB14084:
.LBB14085:
.LBB14086:
.LBB14087:
	.loc 2 402 0
	movl	(%rdi), %eax
	andl	$4088, %eax
	sarl	$3, %eax
	addl	$1, %eax
.LBE14087:
.LBE14086:
	.loc 1 298 0
	cmpl	$3, %eax
	je	.L2278
	.loc 1 301 0
	cmpl	$2, %eax
	je	.L2279
	.loc 1 304 0
	cmpl	$1, %eax
	jne	.L2255
.LVL3528:
.LBB14088:
.LBB14089:
	.loc 13 1863 0
	movl	%edx, 4(%rsp)
.LVL3529:
.LBE14089:
.LBE14088:
	.loc 1 305 0
	movq	%rsp, %rdx
.LVL3530:
.LBB14091:
.LBB14090:
	.loc 13 1863 0
	movl	%ecx, (%rsp)
.LBE14090:
.LBE14091:
	.loc 1 305 0
	call	_Z12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEdd
.LVL3531:
.LBE14085:
.LBE14084:
.LBE14083:
	.loc 1 715 0
	jmp	.L2255
.LVL3532:
	.p2align 4,,10
	.p2align 3
.L2276:
.LBB14104:
.LBB14081:
	.loc 1 620 0
	movq	%rsi, %rdx
.LVL3533:
	movq	%rdi, %rsi
.LVL3534:
	movq	%rcx, %rdi
.LVL3535:
	call	_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_
.LVL3536:
.LBE14081:
.LBE14104:
.LBE14109:
	.loc 1 720 0
	jmp	.L2255
.LVL3537:
	.p2align 4,,10
	.p2align 3
.L2274:
.LBB14110:
.LBB14105:
.LBB14082:
	.loc 1 614 0
	movq	%rsi, %rdx
.LVL3538:
	movq	%rdi, %rsi
.LVL3539:
	movq	%rcx, %rdi
.LVL3540:
	call	_ZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_
.LVL3541:
	jmp	.L2255
.LVL3542:
	.p2align 4,,10
	.p2align 3
.L2275:
	.loc 1 617 0
	movq	%rsi, %rdx
.LVL3543:
	movq	%rdi, %rsi
.LVL3544:
	movq	%rcx, %rdi
.LVL3545:
	call	_ZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_
.LVL3546:
	jmp	.L2255
.LVL3547:
	.p2align 4,,10
	.p2align 3
.L2278:
.LBE14082:
.LBE14105:
.LBB14106:
.LBB14103:
.LBB14102:
.LBB14092:
.LBB14093:
	.loc 13 1863 0
	movl	%edx, 4(%rsp)
.LVL3548:
.LBE14093:
.LBE14092:
	.loc 1 299 0
	movq	%rsp, %rdx
.LVL3549:
.LBB14095:
.LBB14094:
	.loc 13 1863 0
	movl	%ecx, (%rsp)
.LBE14094:
.LBE14095:
	.loc 1 299 0
	call	_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd
.LVL3550:
	jmp	.L2255
.LVL3551:
	.p2align 4,,10
	.p2align 3
.L2279:
.LBB14096:
.LBB14097:
.LBB14098:
.LBB14099:
	.loc 13 1863 0
	movl	%edx, 4(%rsp)
.LVL3552:
.LBE14099:
.LBE14098:
	.loc 1 302 0
	movq	%rsp, %rdx
.LVL3553:
.LBB14101:
.LBB14100:
	.loc 13 1863 0
	movl	%ecx, (%rsp)
.LBE14100:
.LBE14101:
	.loc 1 302 0
	call	_Z12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd
.LVL3554:
	jmp	.L2255
.LVL3555:
.L2264:
.LBE14097:
.LBE14096:
.LBE14102:
.LBE14103:
.LBE14106:
	.loc 1 717 0 discriminator 3
	movl	$_ZZ21constTimeGaussianBlurRKN2cv3MatERS0_ddPK15fastGaussianIIRE19__PRETTY_FUNCTION__, %ecx
	movl	$717, %edx
.LVL3556:
	movl	$.LC45, %esi
.LVL3557:
	movl	$.LC81, %edi
.LVL3558:
	call	__assert_fail
.LVL3559:
.L2277:
.LBE14110:
	.loc 1 720 0
	call	__stack_chk_fail
.LVL3560:
	.cfi_endproc
.LFE11287:
	.size	_Z21constTimeGaussianBlurRKN2cv3MatERS0_ddPK15fastGaussianIIR, .-_Z21constTimeGaussianBlurRKN2cv3MatERS0_ddPK15fastGaussianIIR
	.section	.text.unlikely
.LCOLDE82:
	.text
.LHOTE82:
	.section	.text.unlikely
.LCOLDB83:
	.section	.text.startup
.LHOTB83:
	.p2align 4,,15
	.type	_GLOBAL__sub_I__Z14myGaussianBlurRKN2cv3MatERS0_NS_5Size_IiEEdd, @function
_GLOBAL__sub_I__Z14myGaussianBlurRKN2cv3MatERS0_NS_5Size_IiEEdd:
.LFB12398:
	.loc 1 966 0
	.cfi_startproc
.LVL3561:
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
.LBB14111:
.LBB14112:
	.file 20 "/usr/include/c++/5/iostream"
	.loc 20 74 0
	movl	$_ZStL8__ioinit, %edi
	call	_ZNSt8ios_base4InitC1Ev
.LVL3562:
	movl	$__dso_handle, %edx
	movl	$_ZStL8__ioinit, %esi
	movl	$_ZNSt8ios_base4InitD1Ev, %edi
.LBE14112:
.LBE14111:
	.loc 1 966 0
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB14114:
.LBB14113:
	.loc 20 74 0
	jmp	__cxa_atexit
.LVL3563:
.LBE14113:
.LBE14114:
	.cfi_endproc
.LFE12398:
	.size	_GLOBAL__sub_I__Z14myGaussianBlurRKN2cv3MatERS0_NS_5Size_IiEEdd, .-_GLOBAL__sub_I__Z14myGaussianBlurRKN2cv3MatERS0_NS_5Size_IiEEdd
	.section	.text.unlikely
.LCOLDE83:
	.section	.text.startup
.LHOTE83:
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I__Z14myGaussianBlurRKN2cv3MatERS0_NS_5Size_IiEEdd
	.section	.rodata
	.align 32
	.type	_ZZ9boxFilterRKN2cv3MatERS0_iiE19__PRETTY_FUNCTION__, @object
	.size	_ZZ9boxFilterRKN2cv3MatERS0_iiE19__PRETTY_FUNCTION__, 51
_ZZ9boxFilterRKN2cv3MatERS0_iiE19__PRETTY_FUNCTION__:
	.string	"void boxFilter(const cv::Mat&, cv::Mat&, int, int)"
	.align 32
	.type	_ZZ21constTimeGaussianBlurRKN2cv3MatERS0_ddPK15fastGaussianIIRE19__PRETTY_FUNCTION__, @object
	.size	_ZZ21constTimeGaussianBlurRKN2cv3MatERS0_ddPK15fastGaussianIIRE19__PRETTY_FUNCTION__, 93
_ZZ21constTimeGaussianBlurRKN2cv3MatERS0_ddPK15fastGaussianIIRE19__PRETTY_FUNCTION__:
	.string	"void constTimeGaussianBlur(const cv::Mat&, cv::Mat&, double, double, const fastGaussianIIR*)"
	.align 32
	.type	_ZZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_E19__PRETTY_FUNCTION__, @object
	.size	_ZZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_E19__PRETTY_FUNCTION__, 141
_ZZN15fastGaussianIIR6apply_IN2cv3VecIhLi3EEENS2_IfLi3EEELi3EEEvRKNS1_3MatERS5_E19__PRETTY_FUNCTION__:
	.string	"void fastGaussianIIR::apply_(const cv::Mat&, cv::Mat&) [with SRC_TYPE = cv::Vec<unsigned char, 3>; DST_TYPE = cv::Vec<float, 3>; int cn = 3]"
	.align 32
	.type	_ZZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_E19__PRETTY_FUNCTION__, @object
	.size	_ZZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_E19__PRETTY_FUNCTION__, 141
_ZZN15fastGaussianIIR6apply_IN2cv3VecIhLi2EEENS2_IfLi2EEELi2EEEvRKNS1_3MatERS5_E19__PRETTY_FUNCTION__:
	.string	"void fastGaussianIIR::apply_(const cv::Mat&, cv::Mat&) [with SRC_TYPE = cv::Vec<unsigned char, 2>; DST_TYPE = cv::Vec<float, 2>; int cn = 2]"
	.align 32
	.type	_ZZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_E19__PRETTY_FUNCTION__, @object
	.size	_ZZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_E19__PRETTY_FUNCTION__, 117
_ZZN15fastGaussianIIR6apply_IhfLi1EEEvRKN2cv3MatERS2_E19__PRETTY_FUNCTION__:
	.string	"void fastGaussianIIR::apply_(const cv::Mat&, cv::Mat&) [with SRC_TYPE = unsigned char; DST_TYPE = float; int cn = 1]"
	.align 32
	.type	_ZZ12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, @object
	.size	_ZZ12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, 165
_ZZ12gaussianBlurIhfLi1EEvRKN2cv3MatERS1_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__:
	.string	"void gaussianBlur(const cv::Mat&, cv::Mat&, cv::Size, double, double) [with SRC_TYPE = unsigned char; DST_TYPE = float; int channels = 1; cv::Size = cv::Size_<int>]"
	.align 32
	.type	_ZZ12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, @object
	.size	_ZZ12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, 189
_ZZ12gaussianBlurIN2cv3VecIhLi2EEENS1_IfLi2EEELi2EEvRKNS0_3MatERS4_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__:
	.string	"void gaussianBlur(const cv::Mat&, cv::Mat&, cv::Size, double, double) [with SRC_TYPE = cv::Vec<unsigned char, 2>; DST_TYPE = cv::Vec<float, 2>; int channels = 2; cv::Size = cv::Size_<int>]"
	.align 32
	.type	_ZZ12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, @object
	.size	_ZZ12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__, 189
_ZZ12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEddE19__PRETTY_FUNCTION__:
	.string	"void gaussianBlur(const cv::Mat&, cv::Mat&, cv::Size, double, double) [with SRC_TYPE = cv::Vec<unsigned char, 3>; DST_TYPE = cv::Vec<float, 3>; int channels = 3; cv::Size = cv::Size_<int>]"
	.globl	_ZN15fastGaussianIIR8deriche4E
	.align 32
	.type	_ZN15fastGaussianIIR8deriche4E, @object
	.size	_ZN15fastGaussianIIR8deriche4E, 64
_ZN15fastGaussianIIR8deriche4E:
	.long	2950187609
	.long	1073373218
	.long	3229364607
	.long	-1075526381
	.long	4133240131
	.long	1074576916
	.long	3882188641
	.long	-1076987035
	.long	2585611544
	.long	1071933032
	.long	3950798166
	.long	1073747717
	.long	501102424
	.long	1073490713
	.long	2282553154
	.long	1073434995
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC6:
	.long	0
	.long	1081073664
	.align 8
.LC15:
	.long	0
	.long	1072693248
	.align 8
.LC21:
	.long	501102424
	.long	-1073992935
	.align 8
.LC22:
	.long	2585611544
	.long	1071933032
	.align 8
.LC23:
	.long	2585611544
	.long	-1075550616
	.align 8
.LC24:
	.long	2950187609
	.long	1072324642
	.align 8
.LC25:
	.long	4133240131
	.long	-1073955308
	.align 8
.LC26:
	.long	4133240131
	.long	1073528340
	.align 8
.LC27:
	.long	2282553154
	.long	-1074048653
	.align 8
.LC28:
	.long	3950798166
	.long	1073747717
	.align 8
.LC29:
	.long	3950798166
	.long	-1073735931
	.align 8
.LC30:
	.long	3229364607
	.long	-1076574957
	.align 8
.LC31:
	.long	3882188641
	.long	1069448037
	.align 8
.LC32:
	.long	3882188641
	.long	-1078035611
	.align 8
.LC33:
	.long	0
	.long	-1074790400
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC34:
	.long	0
	.long	-2147483648
	.long	0
	.long	0
	.section	.rodata.cst8
	.align 8
.LC38:
	.long	536225541
	.long	1074007443
	.section	.rodata.cst16
	.align 16
.LC39:
	.long	2147483648
	.long	0
	.long	0
	.long	0
	.align 16
.LC41:
	.long	4294967295
	.long	2147483647
	.long	0
	.long	0
	.section	.rodata.cst8
	.align 8
.LC42:
	.long	2258709403
	.long	1023837339
	.section	.rodata.cst16
	.align 16
.LC47:
	.long	1124007936
	.long	0
	.long	0
	.long	0
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC60:
	.long	1077936128
	.section	.rodata.cst8
	.align 8
.LC67:
	.long	0
	.long	1074266112
	.text
.Letext0:
	.section	.text.unlikely._Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd._omp_fn.0,"axG",@progbits,_Z12gaussianBlurIN2cv3VecIhLi3EEENS1_IfLi3EEELi3EEvRKNS0_3MatERS4_NS0_5Size_IiEEdd,comdat
.Letext_cold0:
	.file 21 "/usr/include/c++/5/bits/stringfwd.h"
	.file 22 "/usr/include/c++/5/cstdio"
	.file 23 "/usr/include/c++/5/cwchar"
	.file 24 "/usr/include/x86_64-linux-gnu/c++/5/bits/c++config.h"
	.file 25 "/usr/include/c++/5/bits/exception_ptr.h"
	.file 26 "/usr/include/c++/5/bits/cpp_type_traits.h"
	.file 27 "/usr/include/c++/5/type_traits"
	.file 28 "/usr/include/c++/5/bits/stl_pair.h"
	.file 29 "/usr/include/c++/5/bits/stl_iterator_base_types.h"
	.file 30 "/usr/include/c++/5/cstdint"
	.file 31 "/usr/include/c++/5/clocale"
	.file 32 "/usr/include/c++/5/new"
	.file 33 "/usr/include/c++/5/bits/allocator.h"
	.file 34 "/usr/include/c++/5/cstdlib"
	.file 35 "/usr/include/c++/5/bits/alloc_traits.h"
	.file 36 "/usr/include/c++/5/initializer_list"
	.file 37 "/usr/include/c++/5/system_error"
	.file 38 "/usr/include/c++/5/bits/ios_base.h"
	.file 39 "/usr/include/c++/5/cwctype"
	.file 40 "/usr/include/c++/5/bits/ostream.tcc"
	.file 41 "/usr/include/c++/5/cstring"
	.file 42 "/usr/include/c++/5/bits/algorithmfwd.h"
	.file 43 "/usr/include/c++/5/cmath"
	.file 44 "/usr/include/c++/5/debug/debug.h"
	.file 45 "/usr/include/c++/5/bits/uniform_int_dist.h"
	.file 46 "/usr/include/c++/5/bits/random.h"
	.file 47 "/usr/include/c++/5/bits/vector.tcc"
	.file 48 "/usr/include/c++/5/bits/stl_algo.h"
	.file 49 "/usr/include/c++/5/cstddef"
	.file 50 "/usr/include/c++/5/bits/uses_allocator.h"
	.file 51 "/usr/include/c++/5/tuple"
	.file 52 "/usr/include/c++/5/iosfwd"
	.file 53 "/usr/include/c++/5/bits/ptr_traits.h"
	.file 54 "/usr/include/c++/5/bits/basic_ios.h"
	.file 55 "/usr/include/c++/5/bits/move.h"
	.file 56 "/usr/include/c++/5/bits/stl_iterator_base_funcs.h"
	.file 57 "/usr/include/c++/5/bits/functexcept.h"
	.file 58 "/usr/include/c++/5/bits/ostream_insert.h"
	.file 59 "/usr/include/c++/5/bits/predefined_ops.h"
	.file 60 "/usr/include/c++/5/ext/numeric_traits.h"
	.file 61 "/usr/include/c++/5/ext/alloc_traits.h"
	.file 62 "/usr/include/c++/5/bits/stl_iterator.h"
	.file 63 "/usr/include/c++/5/ext/type_traits.h"
	.file 64 "/usr/lib/gcc/x86_64-linux-gnu/5/include/stddef.h"
	.file 65 "/usr/include/x86_64-linux-gnu/bits/types.h"
	.file 66 "/usr/include/stdio.h"
	.file 67 "/usr/include/libio.h"
	.file 68 "/usr/include/wchar.h"
	.file 69 "/usr/include/_G_config.h"
	.file 70 "<built-in>"
	.file 71 "/usr/include/x86_64-linux-gnu/bits/stdio.h"
	.file 72 "/usr/include/x86_64-linux-gnu/bits/wchar2.h"
	.file 73 "/usr/include/time.h"
	.file 74 "/usr/include/stdint.h"
	.file 75 "/usr/include/locale.h"
	.file 76 "/usr/include/x86_64-linux-gnu/c++/5/bits/atomic_word.h"
	.file 77 "/usr/include/stdlib.h"
	.file 78 "/usr/include/x86_64-linux-gnu/bits/stdlib-bsearch.h"
	.file 79 "/usr/include/x86_64-linux-gnu/bits/stdlib.h"
	.file 80 "/usr/include/wctype.h"
	.file 81 "/usr/include/string.h"
	.file 82 "/usr/include/x86_64-linux-gnu/bits/mathdef.h"
	.file 83 "/home/xiaocen/Software/opencv/include/opencv2/core/types_c.h"
	.file 84 "/home/xiaocen/Software/opencv/include/opencv2/core/core.hpp"
	.file 85 "/home/xiaocen/Software/opencv/include/opencv2/imgproc/imgproc.hpp"
	.file 86 "/home/xiaocen/Software/opencv/include/opencv2/flann/miniflann.hpp"
	.file 87 "/home/xiaocen/Software/opencv/include/opencv2/objdetect/objdetect.hpp"
	.file 88 "/home/xiaocen/Software/opencv/include/opencv2/calib3d/calib3d.hpp"
	.file 89 "/home/xiaocen/Software/opencv/include/opencv2/contrib/openfabmap.hpp"
	.file 90 "/home/xiaocen/Software/opencv/include/opencv2/highgui/highgui.hpp"
	.file 91 "/home/xiaocen/Software/opencv/include/opencv2/flann/defines.h"
	.file 92 "/home/xiaocen/Software/lapack/lapack_build/include/lapacke.h"
	.file 93 "/usr/include/assert.h"
	.file 94 "/usr/lib/gcc/x86_64-linux-gnu/5/include/omp.h"
	.file 95 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h"
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	0x2e4f1
	.value	0x4
	.long	.Ldebug_abbrev0
	.byte	0x8
	.uleb128 0x1
	.long	.LASF3068
	.byte	0x4
	.long	.LASF3069
	.long	.LASF3070
	.long	.Ldebug_ranges0+0xd040
	.quad	0
	.long	.Ldebug_line0
	.uleb128 0x2
	.byte	0x8
	.byte	0x4
	.long	.LASF0
	.uleb128 0x3
	.byte	0x4
	.byte	0x5
	.string	"int"
	.uleb128 0x4
	.string	"std"
	.byte	0x46
	.byte	0
	.long	0x7965
	.uleb128 0x5
	.long	.LASF1
	.byte	0x18
	.byte	0xda
	.long	0x1a36
	.uleb128 0x6
	.long	.LASF264
	.byte	0x20
	.byte	0x9
	.byte	0x47
	.long	0x1a20
	.uleb128 0x7
	.long	.LASF19
	.byte	0x8
	.byte	0x9
	.byte	0x6a
	.long	0xb3
	.uleb128 0x8
	.long	0x22c8
	.byte	0
	.uleb128 0x9
	.long	.LASF6
	.byte	0x9
	.byte	0x6f
	.long	0xb3
	.byte	0
	.uleb128 0xa
	.long	.LASF19
	.byte	0x9
	.byte	0x6c
	.long	.LASF21
	.long	0x8a
	.long	0x9a
	.uleb128 0xb
	.long	0xa5e9
	.uleb128 0xc
	.long	0xb3
	.uleb128 0xc
	.long	0xa220
	.byte	0
	.uleb128 0xd
	.long	.LASF447
	.long	.LASF697
	.long	0xa7
	.uleb128 0xb
	.long	0xa5e9
	.uleb128 0xb
	.long	0x30
	.byte	0
	.byte	0
	.uleb128 0xe
	.long	.LASF4
	.byte	0x9
	.byte	0x56
	.long	0x7bcc
	.byte	0x1
	.uleb128 0xf
	.byte	0x4
	.long	0x913b
	.byte	0x9
	.byte	0x75
	.long	0xd2
	.uleb128 0x10
	.long	.LASF287
	.byte	0xf
	.byte	0
	.uleb128 0x11
	.byte	0x10
	.byte	0x9
	.byte	0x78
	.long	0xf1
	.uleb128 0x12
	.long	.LASF2
	.byte	0x9
	.byte	0x79
	.long	0xa5ef
	.uleb128 0x12
	.long	.LASF3
	.byte	0x9
	.byte	0x7a
	.long	0xf1
	.byte	0
	.uleb128 0xe
	.long	.LASF5
	.byte	0x9
	.byte	0x52
	.long	0x7be2
	.byte	0x1
	.uleb128 0x13
	.long	.LASF455
	.byte	0x9
	.byte	0x5f
	.long	0x109
	.byte	0x1
	.uleb128 0x14
	.long	0xf1
	.uleb128 0x9
	.long	.LASF7
	.byte	0x9
	.byte	0x72
	.long	0x59
	.byte	0
	.uleb128 0x9
	.long	.LASF8
	.byte	0x9
	.byte	0x73
	.long	0xf1
	.byte	0x8
	.uleb128 0x15
	.long	0xd2
	.byte	0x10
	.uleb128 0x16
	.long	.LASF17
	.byte	0x9
	.byte	0x4a
	.long	0x7ca1
	.uleb128 0xe
	.long	.LASF9
	.byte	0x9
	.byte	0x51
	.long	0x12c
	.byte	0x1
	.uleb128 0xe
	.long	.LASF10
	.byte	0x9
	.byte	0x54
	.long	0x7bed
	.byte	0x1
	.uleb128 0xe
	.long	.LASF11
	.byte	0x9
	.byte	0x55
	.long	0x7bf8
	.byte	0x1
	.uleb128 0xe
	.long	.LASF12
	.byte	0x9
	.byte	0x57
	.long	0x7bd7
	.byte	0x1
	.uleb128 0xe
	.long	.LASF13
	.byte	0x9
	.byte	0x58
	.long	0x7cc0
	.byte	0x1
	.uleb128 0xe
	.long	.LASF14
	.byte	0x9
	.byte	0x5a
	.long	0x7ee7
	.byte	0x1
	.uleb128 0xe
	.long	.LASF15
	.byte	0x9
	.byte	0x5b
	.long	0x252e
	.byte	0x1
	.uleb128 0xe
	.long	.LASF16
	.byte	0x9
	.byte	0x5c
	.long	0x2533
	.byte	0x1
	.uleb128 0x16
	.long	.LASF18
	.byte	0x9
	.byte	0x66
	.long	0x173
	.uleb128 0xa
	.long	.LASF20
	.byte	0x9
	.byte	0x7e
	.long	.LASF22
	.long	0x1b5
	.long	0x1c0
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xb3
	.byte	0
	.uleb128 0xa
	.long	.LASF23
	.byte	0x9
	.byte	0x82
	.long	.LASF24
	.long	0x1d3
	.long	0x1de
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x17
	.long	.LASF20
	.byte	0x9
	.byte	0x86
	.long	.LASF26
	.long	0xb3
	.long	0x1f5
	.long	0x1fb
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x17
	.long	.LASF25
	.byte	0x9
	.byte	0x8a
	.long	.LASF27
	.long	0xb3
	.long	0x212
	.long	0x218
	.uleb128 0xb
	.long	0xa5ff
	.byte	0
	.uleb128 0x17
	.long	.LASF25
	.byte	0x9
	.byte	0x94
	.long	.LASF28
	.long	0x15b
	.long	0x22f
	.long	0x235
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0xa
	.long	.LASF29
	.byte	0x9
	.byte	0x9e
	.long	.LASF30
	.long	0x248
	.long	0x253
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0xa
	.long	.LASF31
	.byte	0x9
	.byte	0xa2
	.long	.LASF32
	.long	0x266
	.long	0x271
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x17
	.long	.LASF33
	.byte	0x9
	.byte	0xa9
	.long	.LASF34
	.long	0x9ef1
	.long	0x288
	.long	0x28e
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x17
	.long	.LASF35
	.byte	0x9
	.byte	0xae
	.long	.LASF36
	.long	0xb3
	.long	0x2a5
	.long	0x2b5
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa60b
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0xa
	.long	.LASF37
	.byte	0x9
	.byte	0xb1
	.long	.LASF38
	.long	0x2c8
	.long	0x2ce
	.uleb128 0xb
	.long	0xa5ff
	.byte	0
	.uleb128 0xa
	.long	.LASF39
	.byte	0x9
	.byte	0xb8
	.long	.LASF40
	.long	0x2e1
	.long	0x2ec
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0xa
	.long	.LASF41
	.byte	0x9
	.byte	0xce
	.long	.LASF42
	.long	0x2ff
	.long	0x30f
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0xa
	.long	.LASF43
	.byte	0x9
	.byte	0xe7
	.long	.LASF44
	.long	0x322
	.long	0x332
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0x17
	.long	.LASF45
	.byte	0x9
	.byte	0xea
	.long	.LASF46
	.long	0xa611
	.long	0x349
	.long	0x34f
	.uleb128 0xb
	.long	0xa5ff
	.byte	0
	.uleb128 0x17
	.long	.LASF45
	.byte	0x9
	.byte	0xee
	.long	.LASF47
	.long	0xa617
	.long	0x366
	.long	0x36c
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x14
	.long	0x137
	.uleb128 0x18
	.long	.LASF48
	.byte	0x9
	.value	0x102
	.long	.LASF51
	.long	0xf1
	.long	0x389
	.long	0x399
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x19
	.long	.LASF49
	.byte	0x9
	.value	0x10c
	.long	.LASF67
	.long	0x3ad
	.long	0x3c2
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x18
	.long	.LASF50
	.byte	0x9
	.value	0x115
	.long	.LASF52
	.long	0xf1
	.long	0x3da
	.long	0x3ea
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x18
	.long	.LASF53
	.byte	0x9
	.value	0x11d
	.long	.LASF54
	.long	0x9ef1
	.long	0x402
	.long	0x40d
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1a
	.long	.LASF55
	.byte	0x9
	.value	0x126
	.long	.LASF57
	.long	0x42d
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1a
	.long	.LASF56
	.byte	0x9
	.value	0x12f
	.long	.LASF58
	.long	0x44d
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1a
	.long	.LASF59
	.byte	0x9
	.value	0x138
	.long	.LASF60
	.long	0x46d
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0x1a
	.long	.LASF61
	.byte	0x9
	.value	0x14b
	.long	.LASF62
	.long	0x48d
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x167
	.uleb128 0xc
	.long	0x167
	.byte	0
	.uleb128 0x1a
	.long	.LASF61
	.byte	0x9
	.value	0x14f
	.long	.LASF63
	.long	0x4ad
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x173
	.uleb128 0xc
	.long	0x173
	.byte	0
	.uleb128 0x1a
	.long	.LASF61
	.byte	0x9
	.value	0x154
	.long	.LASF64
	.long	0x4cd
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x919c
	.byte	0
	.uleb128 0x1a
	.long	.LASF61
	.byte	0x9
	.value	0x158
	.long	.LASF65
	.long	0x4ed
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1b
	.long	.LASF66
	.byte	0x9
	.value	0x15d
	.long	.LASF68
	.long	0x30
	.long	0x50c
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x19
	.long	.LASF69
	.byte	0x9
	.value	0x16a
	.long	.LASF70
	.long	0x520
	.long	0x52b
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa61d
	.byte	0
	.uleb128 0x19
	.long	.LASF71
	.byte	0x9
	.value	0x16d
	.long	.LASF72
	.long	0x53f
	.long	0x559
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x19
	.long	.LASF73
	.byte	0x9
	.value	0x171
	.long	.LASF74
	.long	0x56d
	.long	0x57d
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1c
	.long	.LASF75
	.byte	0x9
	.value	0x17b
	.long	.LASF76
	.byte	0x1
	.long	0x592
	.long	0x598
	.uleb128 0xb
	.long	0xa5ff
	.byte	0
	.uleb128 0x1d
	.long	.LASF75
	.byte	0x9
	.value	0x186
	.long	.LASF90
	.byte	0x1
	.long	0x5ad
	.long	0x5b8
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa220
	.byte	0
	.uleb128 0x1c
	.long	.LASF75
	.byte	0x9
	.value	0x18e
	.long	.LASF77
	.byte	0x1
	.long	0x5cd
	.long	0x5d8
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa61d
	.byte	0
	.uleb128 0x1c
	.long	.LASF75
	.byte	0x9
	.value	0x19a
	.long	.LASF78
	.byte	0x1
	.long	0x5ed
	.long	0x602
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa61d
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1c
	.long	.LASF75
	.byte	0x9
	.value	0x1aa
	.long	.LASF79
	.byte	0x1
	.long	0x617
	.long	0x631
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa61d
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xa220
	.byte	0
	.uleb128 0x1c
	.long	.LASF75
	.byte	0x9
	.value	0x1bc
	.long	.LASF80
	.byte	0x1
	.long	0x646
	.long	0x65b
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xa220
	.byte	0
	.uleb128 0x1c
	.long	.LASF75
	.byte	0x9
	.value	0x1c6
	.long	.LASF81
	.byte	0x1
	.long	0x670
	.long	0x680
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xa220
	.byte	0
	.uleb128 0x1c
	.long	.LASF75
	.byte	0x9
	.value	0x1d0
	.long	.LASF82
	.byte	0x1
	.long	0x695
	.long	0x6aa
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x91a2
	.uleb128 0xc
	.long	0xa220
	.byte	0
	.uleb128 0x1c
	.long	.LASF75
	.byte	0x9
	.value	0x1dc
	.long	.LASF83
	.byte	0x1
	.long	0x6bf
	.long	0x6ca
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa623
	.byte	0
	.uleb128 0x1c
	.long	.LASF75
	.byte	0x9
	.value	0x1f7
	.long	.LASF84
	.byte	0x1
	.long	0x6df
	.long	0x6ef
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x2538
	.uleb128 0xc
	.long	0xa220
	.byte	0
	.uleb128 0x1c
	.long	.LASF75
	.byte	0x9
	.value	0x1fb
	.long	.LASF85
	.byte	0x1
	.long	0x704
	.long	0x714
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa61d
	.uleb128 0xc
	.long	0xa220
	.byte	0
	.uleb128 0x1c
	.long	.LASF75
	.byte	0x9
	.value	0x1ff
	.long	.LASF86
	.byte	0x1
	.long	0x729
	.long	0x739
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa623
	.uleb128 0xc
	.long	0xa220
	.byte	0
	.uleb128 0x1c
	.long	.LASF87
	.byte	0x9
	.value	0x21e
	.long	.LASF88
	.byte	0x1
	.long	0x74e
	.long	0x759
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0x9
	.value	0x226
	.long	.LASF91
	.long	0xa629
	.byte	0x1
	.long	0x772
	.long	0x77d
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa61d
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0x9
	.value	0x22e
	.long	.LASF92
	.long	0xa629
	.byte	0x1
	.long	0x796
	.long	0x7a1
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0x9
	.value	0x239
	.long	.LASF93
	.long	0xa629
	.byte	0x1
	.long	0x7ba
	.long	0x7c5
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0x9
	.value	0x24b
	.long	.LASF94
	.long	0xa629
	.byte	0x1
	.long	0x7de
	.long	0x7e9
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa623
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0x9
	.value	0x256
	.long	.LASF95
	.long	0xa629
	.byte	0x1
	.long	0x802
	.long	0x80d
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x2538
	.byte	0
	.uleb128 0x1e
	.long	.LASF96
	.byte	0x9
	.value	0x263
	.long	.LASF97
	.long	0x167
	.byte	0x1
	.long	0x826
	.long	0x82c
	.uleb128 0xb
	.long	0xa5ff
	.byte	0
	.uleb128 0x1e
	.long	.LASF96
	.byte	0x9
	.value	0x26b
	.long	.LASF98
	.long	0x173
	.byte	0x1
	.long	0x845
	.long	0x84b
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1f
	.string	"end"
	.byte	0x9
	.value	0x273
	.long	.LASF99
	.long	0x167
	.byte	0x1
	.long	0x864
	.long	0x86a
	.uleb128 0xb
	.long	0xa5ff
	.byte	0
	.uleb128 0x1f
	.string	"end"
	.byte	0x9
	.value	0x27b
	.long	.LASF100
	.long	0x173
	.byte	0x1
	.long	0x883
	.long	0x889
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF101
	.byte	0x9
	.value	0x284
	.long	.LASF102
	.long	0x18b
	.byte	0x1
	.long	0x8a2
	.long	0x8a8
	.uleb128 0xb
	.long	0xa5ff
	.byte	0
	.uleb128 0x1e
	.long	.LASF101
	.byte	0x9
	.value	0x28d
	.long	.LASF103
	.long	0x17f
	.byte	0x1
	.long	0x8c1
	.long	0x8c7
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF104
	.byte	0x9
	.value	0x296
	.long	.LASF105
	.long	0x18b
	.byte	0x1
	.long	0x8e0
	.long	0x8e6
	.uleb128 0xb
	.long	0xa5ff
	.byte	0
	.uleb128 0x1e
	.long	.LASF104
	.byte	0x9
	.value	0x29f
	.long	.LASF106
	.long	0x17f
	.byte	0x1
	.long	0x8ff
	.long	0x905
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF107
	.byte	0x9
	.value	0x2a8
	.long	.LASF108
	.long	0x173
	.byte	0x1
	.long	0x91e
	.long	0x924
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF109
	.byte	0x9
	.value	0x2b0
	.long	.LASF110
	.long	0x173
	.byte	0x1
	.long	0x93d
	.long	0x943
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF111
	.byte	0x9
	.value	0x2b9
	.long	.LASF112
	.long	0x17f
	.byte	0x1
	.long	0x95c
	.long	0x962
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF113
	.byte	0x9
	.value	0x2c2
	.long	.LASF114
	.long	0x17f
	.byte	0x1
	.long	0x97b
	.long	0x981
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF115
	.byte	0x9
	.value	0x2cb
	.long	.LASF116
	.long	0xf1
	.byte	0x1
	.long	0x99a
	.long	0x9a0
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF117
	.byte	0x9
	.value	0x2d1
	.long	.LASF118
	.long	0xf1
	.byte	0x1
	.long	0x9b9
	.long	0x9bf
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF119
	.byte	0x9
	.value	0x2d6
	.long	.LASF120
	.long	0xf1
	.byte	0x1
	.long	0x9d8
	.long	0x9de
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1c
	.long	.LASF121
	.byte	0x9
	.value	0x2e4
	.long	.LASF122
	.byte	0x1
	.long	0x9f3
	.long	0xa03
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0x1c
	.long	.LASF121
	.byte	0x9
	.value	0x2f1
	.long	.LASF123
	.byte	0x1
	.long	0xa18
	.long	0xa23
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1c
	.long	.LASF124
	.byte	0x9
	.value	0x2f7
	.long	.LASF125
	.byte	0x1
	.long	0xa38
	.long	0xa3e
	.uleb128 0xb
	.long	0xa5ff
	.byte	0
	.uleb128 0x1e
	.long	.LASF126
	.byte	0x9
	.value	0x30a
	.long	.LASF127
	.long	0xf1
	.byte	0x1
	.long	0xa57
	.long	0xa5d
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1c
	.long	.LASF128
	.byte	0x9
	.value	0x322
	.long	.LASF129
	.byte	0x1
	.long	0xa72
	.long	0xa7d
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1c
	.long	.LASF130
	.byte	0x9
	.value	0x328
	.long	.LASF131
	.byte	0x1
	.long	0xa92
	.long	0xa98
	.uleb128 0xb
	.long	0xa5ff
	.byte	0
	.uleb128 0x1e
	.long	.LASF132
	.byte	0x9
	.value	0x330
	.long	.LASF133
	.long	0x9ef1
	.byte	0x1
	.long	0xab1
	.long	0xab7
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x9
	.value	0x33f
	.long	.LASF135
	.long	0x14f
	.byte	0x1
	.long	0xad0
	.long	0xadb
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x9
	.value	0x350
	.long	.LASF136
	.long	0x143
	.byte	0x1
	.long	0xaf4
	.long	0xaff
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1f
	.string	"at"
	.byte	0x9
	.value	0x365
	.long	.LASF137
	.long	0x14f
	.byte	0x1
	.long	0xb17
	.long	0xb22
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1f
	.string	"at"
	.byte	0x9
	.value	0x37a
	.long	.LASF138
	.long	0x143
	.byte	0x1
	.long	0xb3a
	.long	0xb45
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF139
	.byte	0x9
	.value	0x38a
	.long	.LASF140
	.long	0x143
	.byte	0x1
	.long	0xb5e
	.long	0xb64
	.uleb128 0xb
	.long	0xa5ff
	.byte	0
	.uleb128 0x1e
	.long	.LASF139
	.byte	0x9
	.value	0x392
	.long	.LASF141
	.long	0x14f
	.byte	0x1
	.long	0xb7d
	.long	0xb83
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF142
	.byte	0x9
	.value	0x39a
	.long	.LASF143
	.long	0x143
	.byte	0x1
	.long	0xb9c
	.long	0xba2
	.uleb128 0xb
	.long	0xa5ff
	.byte	0
	.uleb128 0x1e
	.long	.LASF142
	.byte	0x9
	.value	0x3a2
	.long	.LASF144
	.long	0x14f
	.byte	0x1
	.long	0xbbb
	.long	0xbc1
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF145
	.byte	0x9
	.value	0x3ad
	.long	.LASF146
	.long	0xa629
	.byte	0x1
	.long	0xbda
	.long	0xbe5
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa61d
	.byte	0
	.uleb128 0x1e
	.long	.LASF145
	.byte	0x9
	.value	0x3b6
	.long	.LASF147
	.long	0xa629
	.byte	0x1
	.long	0xbfe
	.long	0xc09
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1e
	.long	.LASF145
	.byte	0x9
	.value	0x3bf
	.long	.LASF148
	.long	0xa629
	.byte	0x1
	.long	0xc22
	.long	0xc2d
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0x1e
	.long	.LASF145
	.byte	0x9
	.value	0x3cc
	.long	.LASF149
	.long	0xa629
	.byte	0x1
	.long	0xc46
	.long	0xc51
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x2538
	.byte	0
	.uleb128 0x1e
	.long	.LASF150
	.byte	0x9
	.value	0x3d6
	.long	.LASF151
	.long	0xa629
	.byte	0x1
	.long	0xc6a
	.long	0xc75
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa61d
	.byte	0
	.uleb128 0x1e
	.long	.LASF150
	.byte	0x9
	.value	0x3e7
	.long	.LASF152
	.long	0xa629
	.byte	0x1
	.long	0xc8e
	.long	0xca3
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa61d
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF150
	.byte	0x9
	.value	0x3f3
	.long	.LASF153
	.long	0xa629
	.byte	0x1
	.long	0xcbc
	.long	0xccc
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF150
	.byte	0x9
	.value	0x400
	.long	.LASF154
	.long	0xa629
	.byte	0x1
	.long	0xce5
	.long	0xcf0
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1e
	.long	.LASF150
	.byte	0x9
	.value	0x411
	.long	.LASF155
	.long	0xa629
	.byte	0x1
	.long	0xd09
	.long	0xd19
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0x1e
	.long	.LASF150
	.byte	0x9
	.value	0x41b
	.long	.LASF156
	.long	0xa629
	.byte	0x1
	.long	0xd32
	.long	0xd3d
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x2538
	.byte	0
	.uleb128 0x1c
	.long	.LASF157
	.byte	0x9
	.value	0x436
	.long	.LASF158
	.byte	0x1
	.long	0xd52
	.long	0xd5d
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0x1e
	.long	.LASF159
	.byte	0x9
	.value	0x445
	.long	.LASF160
	.long	0xa629
	.byte	0x1
	.long	0xd76
	.long	0xd81
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa61d
	.byte	0
	.uleb128 0x1e
	.long	.LASF159
	.byte	0x9
	.value	0x455
	.long	.LASF161
	.long	0xa629
	.byte	0x1
	.long	0xd9a
	.long	0xda5
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa623
	.byte	0
	.uleb128 0x1e
	.long	.LASF159
	.byte	0x9
	.value	0x46b
	.long	.LASF162
	.long	0xa629
	.byte	0x1
	.long	0xdbe
	.long	0xdd3
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa61d
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF159
	.byte	0x9
	.value	0x47b
	.long	.LASF163
	.long	0xa629
	.byte	0x1
	.long	0xdec
	.long	0xdfc
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF159
	.byte	0x9
	.value	0x48b
	.long	.LASF164
	.long	0xa629
	.byte	0x1
	.long	0xe15
	.long	0xe20
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1e
	.long	.LASF159
	.byte	0x9
	.value	0x49c
	.long	.LASF165
	.long	0xa629
	.byte	0x1
	.long	0xe39
	.long	0xe49
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0x1e
	.long	.LASF159
	.byte	0x9
	.value	0x4b8
	.long	.LASF166
	.long	0xa629
	.byte	0x1
	.long	0xe62
	.long	0xe6d
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x2538
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0x9
	.value	0x4cd
	.long	.LASF168
	.long	0x167
	.byte	0x1
	.long	0xe86
	.long	0xe9b
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x173
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0x1c
	.long	.LASF167
	.byte	0x9
	.value	0x51b
	.long	.LASF169
	.byte	0x1
	.long	0xeb0
	.long	0xec0
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x167
	.uleb128 0xc
	.long	0x2538
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0x9
	.value	0x52f
	.long	.LASF170
	.long	0xa629
	.byte	0x1
	.long	0xed9
	.long	0xee9
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xa61d
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0x9
	.value	0x546
	.long	.LASF171
	.long	0xa629
	.byte	0x1
	.long	0xf02
	.long	0xf1c
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xa61d
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0x9
	.value	0x55d
	.long	.LASF172
	.long	0xa629
	.byte	0x1
	.long	0xf35
	.long	0xf4a
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0x9
	.value	0x570
	.long	.LASF173
	.long	0xa629
	.byte	0x1
	.long	0xf63
	.long	0xf73
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0x9
	.value	0x588
	.long	.LASF174
	.long	0xa629
	.byte	0x1
	.long	0xf8c
	.long	0xfa1
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0x9
	.value	0x59a
	.long	.LASF175
	.long	0x167
	.byte	0x1
	.long	0xfba
	.long	0xfca
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0x1e
	.long	.LASF176
	.byte	0x9
	.value	0x5b2
	.long	.LASF177
	.long	0xa629
	.byte	0x1
	.long	0xfe3
	.long	0xff3
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF176
	.byte	0x9
	.value	0x5c2
	.long	.LASF178
	.long	0x167
	.byte	0x1
	.long	0x100c
	.long	0x1017
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x197
	.byte	0
	.uleb128 0x1e
	.long	.LASF176
	.byte	0x9
	.value	0x5d5
	.long	.LASF179
	.long	0x167
	.byte	0x1
	.long	0x1030
	.long	0x1040
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x197
	.byte	0
	.uleb128 0x1c
	.long	.LASF180
	.byte	0x9
	.value	0x5e5
	.long	.LASF181
	.byte	0x1
	.long	0x1055
	.long	0x105b
	.uleb128 0xb
	.long	0xa5ff
	.byte	0
	.uleb128 0x1e
	.long	.LASF182
	.byte	0x9
	.value	0x5fb
	.long	.LASF183
	.long	0xa629
	.byte	0x1
	.long	0x1074
	.long	0x1089
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xa61d
	.byte	0
	.uleb128 0x1e
	.long	.LASF182
	.byte	0x9
	.value	0x611
	.long	.LASF184
	.long	0xa629
	.byte	0x1
	.long	0x10a2
	.long	0x10c1
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xa61d
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF182
	.byte	0x9
	.value	0x62a
	.long	.LASF185
	.long	0xa629
	.byte	0x1
	.long	0x10da
	.long	0x10f4
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF182
	.byte	0x9
	.value	0x643
	.long	.LASF186
	.long	0xa629
	.byte	0x1
	.long	0x110d
	.long	0x1122
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1e
	.long	.LASF182
	.byte	0x9
	.value	0x65b
	.long	.LASF187
	.long	0xa629
	.byte	0x1
	.long	0x113b
	.long	0x1155
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0x1e
	.long	.LASF182
	.byte	0x9
	.value	0x66d
	.long	.LASF188
	.long	0xa629
	.byte	0x1
	.long	0x116e
	.long	0x1183
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0xa61d
	.byte	0
	.uleb128 0x1e
	.long	.LASF182
	.byte	0x9
	.value	0x681
	.long	.LASF189
	.long	0xa629
	.byte	0x1
	.long	0x119c
	.long	0x11b6
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF182
	.byte	0x9
	.value	0x697
	.long	.LASF190
	.long	0xa629
	.byte	0x1
	.long	0x11cf
	.long	0x11e4
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1e
	.long	.LASF182
	.byte	0x9
	.value	0x6ac
	.long	.LASF191
	.long	0xa629
	.byte	0x1
	.long	0x11fd
	.long	0x1217
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0x1e
	.long	.LASF182
	.byte	0x9
	.value	0x6e5
	.long	.LASF192
	.long	0xa629
	.byte	0x1
	.long	0x1230
	.long	0x124a
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x919c
	.byte	0
	.uleb128 0x1e
	.long	.LASF182
	.byte	0x9
	.value	0x6f0
	.long	.LASF193
	.long	0xa629
	.byte	0x1
	.long	0x1263
	.long	0x127d
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1e
	.long	.LASF182
	.byte	0x9
	.value	0x6fb
	.long	.LASF194
	.long	0xa629
	.byte	0x1
	.long	0x1296
	.long	0x12b0
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x167
	.uleb128 0xc
	.long	0x167
	.byte	0
	.uleb128 0x1e
	.long	.LASF182
	.byte	0x9
	.value	0x706
	.long	.LASF195
	.long	0xa629
	.byte	0x1
	.long	0x12c9
	.long	0x12e3
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x197
	.uleb128 0xc
	.long	0x173
	.uleb128 0xc
	.long	0x173
	.byte	0
	.uleb128 0x1e
	.long	.LASF182
	.byte	0x9
	.value	0x71f
	.long	.LASF196
	.long	0xa629
	.byte	0x1
	.long	0x12fc
	.long	0x1311
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x173
	.uleb128 0xc
	.long	0x173
	.uleb128 0xc
	.long	0x2538
	.byte	0
	.uleb128 0x18
	.long	.LASF197
	.byte	0x9
	.value	0x732
	.long	.LASF198
	.long	0xa629
	.long	0x1329
	.long	0x1343
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x91a2
	.byte	0
	.uleb128 0x18
	.long	.LASF199
	.byte	0x9
	.value	0x736
	.long	.LASF200
	.long	0xa629
	.long	0x135b
	.long	0x1375
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x18
	.long	.LASF201
	.byte	0x9
	.value	0x73a
	.long	.LASF202
	.long	0xa629
	.long	0x138d
	.long	0x139d
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF203
	.byte	0x9
	.value	0x74b
	.long	.LASF204
	.long	0xf1
	.byte	0x1
	.long	0x13b6
	.long	0x13cb
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1c
	.long	.LASF205
	.byte	0x9
	.value	0x755
	.long	.LASF206
	.byte	0x1
	.long	0x13e0
	.long	0x13eb
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0xa629
	.byte	0
	.uleb128 0x1e
	.long	.LASF207
	.byte	0x9
	.value	0x75f
	.long	.LASF208
	.long	0x9472
	.byte	0x1
	.long	0x1404
	.long	0x140a
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF209
	.byte	0x9
	.value	0x769
	.long	.LASF210
	.long	0x9472
	.byte	0x1
	.long	0x1423
	.long	0x1429
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF211
	.byte	0x9
	.value	0x770
	.long	.LASF212
	.long	0x137
	.byte	0x1
	.long	0x1442
	.long	0x1448
	.uleb128 0xb
	.long	0xa605
	.byte	0
	.uleb128 0x1e
	.long	.LASF213
	.byte	0x9
	.value	0x780
	.long	.LASF214
	.long	0xf1
	.byte	0x1
	.long	0x1461
	.long	0x1476
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF213
	.byte	0x9
	.value	0x78d
	.long	.LASF215
	.long	0xf1
	.byte	0x1
	.long	0x148f
	.long	0x149f
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xa61d
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF213
	.byte	0x9
	.value	0x79c
	.long	.LASF216
	.long	0xf1
	.byte	0x1
	.long	0x14b8
	.long	0x14c8
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF213
	.byte	0x9
	.value	0x7ad
	.long	.LASF217
	.long	0xf1
	.byte	0x1
	.long	0x14e1
	.long	0x14f1
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x91a2
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF218
	.byte	0x9
	.value	0x7ba
	.long	.LASF219
	.long	0xf1
	.byte	0x1
	.long	0x150a
	.long	0x151a
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xa61d
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF218
	.byte	0x9
	.value	0x7cb
	.long	.LASF220
	.long	0xf1
	.byte	0x1
	.long	0x1533
	.long	0x1548
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF218
	.byte	0x9
	.value	0x7d8
	.long	.LASF221
	.long	0xf1
	.byte	0x1
	.long	0x1561
	.long	0x1571
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF218
	.byte	0x9
	.value	0x7e9
	.long	.LASF222
	.long	0xf1
	.byte	0x1
	.long	0x158a
	.long	0x159a
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x91a2
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF223
	.byte	0x9
	.value	0x7f7
	.long	.LASF224
	.long	0xf1
	.byte	0x1
	.long	0x15b3
	.long	0x15c3
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xa61d
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF223
	.byte	0x9
	.value	0x808
	.long	.LASF225
	.long	0xf1
	.byte	0x1
	.long	0x15dc
	.long	0x15f1
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF223
	.byte	0x9
	.value	0x815
	.long	.LASF226
	.long	0xf1
	.byte	0x1
	.long	0x160a
	.long	0x161a
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF223
	.byte	0x9
	.value	0x828
	.long	.LASF227
	.long	0xf1
	.byte	0x1
	.long	0x1633
	.long	0x1643
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x91a2
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF228
	.byte	0x9
	.value	0x837
	.long	.LASF229
	.long	0xf1
	.byte	0x1
	.long	0x165c
	.long	0x166c
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xa61d
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF228
	.byte	0x9
	.value	0x848
	.long	.LASF230
	.long	0xf1
	.byte	0x1
	.long	0x1685
	.long	0x169a
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF228
	.byte	0x9
	.value	0x855
	.long	.LASF231
	.long	0xf1
	.byte	0x1
	.long	0x16b3
	.long	0x16c3
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF228
	.byte	0x9
	.value	0x868
	.long	.LASF232
	.long	0xf1
	.byte	0x1
	.long	0x16dc
	.long	0x16ec
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x91a2
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF233
	.byte	0x9
	.value	0x876
	.long	.LASF234
	.long	0xf1
	.byte	0x1
	.long	0x1705
	.long	0x1715
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xa61d
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF233
	.byte	0x9
	.value	0x887
	.long	.LASF235
	.long	0xf1
	.byte	0x1
	.long	0x172e
	.long	0x1743
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF233
	.byte	0x9
	.value	0x895
	.long	.LASF236
	.long	0xf1
	.byte	0x1
	.long	0x175c
	.long	0x176c
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF233
	.byte	0x9
	.value	0x8a6
	.long	.LASF237
	.long	0xf1
	.byte	0x1
	.long	0x1785
	.long	0x1795
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x91a2
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF238
	.byte	0x9
	.value	0x8b5
	.long	.LASF239
	.long	0xf1
	.byte	0x1
	.long	0x17ae
	.long	0x17be
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xa61d
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF238
	.byte	0x9
	.value	0x8c6
	.long	.LASF240
	.long	0xf1
	.byte	0x1
	.long	0x17d7
	.long	0x17ec
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF238
	.byte	0x9
	.value	0x8d4
	.long	.LASF241
	.long	0xf1
	.byte	0x1
	.long	0x1805
	.long	0x1815
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF238
	.byte	0x9
	.value	0x8e5
	.long	.LASF242
	.long	0xf1
	.byte	0x1
	.long	0x182e
	.long	0x183e
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x91a2
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF243
	.byte	0x9
	.value	0x8f5
	.long	.LASF244
	.long	0x4d
	.byte	0x1
	.long	0x1857
	.long	0x1867
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF245
	.byte	0x9
	.value	0x908
	.long	.LASF246
	.long	0x30
	.byte	0x1
	.long	0x1880
	.long	0x188b
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xa61d
	.byte	0
	.uleb128 0x1e
	.long	.LASF245
	.byte	0x9
	.value	0x928
	.long	.LASF247
	.long	0x30
	.byte	0x1
	.long	0x18a4
	.long	0x18b9
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xa61d
	.byte	0
	.uleb128 0x1e
	.long	.LASF245
	.byte	0x9
	.value	0x942
	.long	.LASF248
	.long	0x30
	.byte	0x1
	.long	0x18d2
	.long	0x18f1
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xa61d
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0x1e
	.long	.LASF245
	.byte	0x9
	.value	0x954
	.long	.LASF249
	.long	0x30
	.byte	0x1
	.long	0x190a
	.long	0x1915
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1e
	.long	.LASF245
	.byte	0x9
	.value	0x96c
	.long	.LASF250
	.long	0x30
	.byte	0x1
	.long	0x192e
	.long	0x1943
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1e
	.long	.LASF245
	.byte	0x9
	.value	0x987
	.long	.LASF251
	.long	0x30
	.byte	0x1
	.long	0x195c
	.long	0x1976
	.uleb128 0xb
	.long	0xa605
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0xf1
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xf1
	.byte	0
	.uleb128 0xa
	.long	.LASF252
	.byte	0xb
	.byte	0xd2
	.long	.LASF253
	.long	0x1992
	.long	0x19a7
	.uleb128 0x20
	.long	.LASF256
	.long	0x9472
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x1fbe
	.byte	0
	.uleb128 0xa
	.long	.LASF254
	.byte	0x9
	.byte	0xbf
	.long	.LASF255
	.long	0x19c3
	.long	0x19d8
	.uleb128 0x20
	.long	.LASF257
	.long	0x9472
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x1ecd
	.byte	0
	.uleb128 0xa
	.long	.LASF252
	.byte	0x9
	.byte	0xd3
	.long	.LASF258
	.long	0x19f4
	.long	0x1a04
	.uleb128 0x20
	.long	.LASF257
	.long	0x9472
	.uleb128 0xb
	.long	0xa5ff
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x20
	.long	.LASF259
	.long	0x91a2
	.uleb128 0x21
	.long	.LASF260
	.long	0x1ffe
	.uleb128 0x21
	.long	.LASF261
	.long	0x22c8
	.byte	0
	.uleb128 0x14
	.long	0x4d
	.uleb128 0x16
	.long	.LASF262
	.byte	0x15
	.byte	0x4a
	.long	0x4d
	.uleb128 0x14
	.long	0x1a25
	.byte	0
	.uleb128 0x22
	.byte	0x18
	.byte	0xda
	.long	0x42
	.uleb128 0x23
	.byte	0x16
	.byte	0x62
	.long	0x91a9
	.uleb128 0x23
	.byte	0x16
	.byte	0x63
	.long	0x947d
	.uleb128 0x23
	.byte	0x16
	.byte	0x65
	.long	0x9488
	.uleb128 0x23
	.byte	0x16
	.byte	0x66
	.long	0x94a0
	.uleb128 0x23
	.byte	0x16
	.byte	0x67
	.long	0x94b5
	.uleb128 0x23
	.byte	0x16
	.byte	0x68
	.long	0x94cb
	.uleb128 0x23
	.byte	0x16
	.byte	0x69
	.long	0x94e1
	.uleb128 0x23
	.byte	0x16
	.byte	0x6a
	.long	0x94f6
	.uleb128 0x23
	.byte	0x16
	.byte	0x6b
	.long	0x950c
	.uleb128 0x23
	.byte	0x16
	.byte	0x6c
	.long	0x952d
	.uleb128 0x23
	.byte	0x16
	.byte	0x6d
	.long	0x954c
	.uleb128 0x23
	.byte	0x16
	.byte	0x71
	.long	0x9567
	.uleb128 0x23
	.byte	0x16
	.byte	0x72
	.long	0x958c
	.uleb128 0x23
	.byte	0x16
	.byte	0x74
	.long	0x95ac
	.uleb128 0x23
	.byte	0x16
	.byte	0x75
	.long	0x95cc
	.uleb128 0x23
	.byte	0x16
	.byte	0x76
	.long	0x95f2
	.uleb128 0x23
	.byte	0x16
	.byte	0x78
	.long	0x9608
	.uleb128 0x23
	.byte	0x16
	.byte	0x79
	.long	0x961e
	.uleb128 0x23
	.byte	0x16
	.byte	0x7c
	.long	0x9629
	.uleb128 0x23
	.byte	0x16
	.byte	0x7e
	.long	0x963f
	.uleb128 0x23
	.byte	0x16
	.byte	0x83
	.long	0x9651
	.uleb128 0x23
	.byte	0x16
	.byte	0x84
	.long	0x9666
	.uleb128 0x23
	.byte	0x16
	.byte	0x85
	.long	0x9680
	.uleb128 0x23
	.byte	0x16
	.byte	0x87
	.long	0x9692
	.uleb128 0x23
	.byte	0x16
	.byte	0x88
	.long	0x96a9
	.uleb128 0x23
	.byte	0x16
	.byte	0x8b
	.long	0x96ce
	.uleb128 0x23
	.byte	0x16
	.byte	0x8d
	.long	0x96d9
	.uleb128 0x23
	.byte	0x16
	.byte	0x8f
	.long	0x96ee
	.uleb128 0x23
	.byte	0x17
	.byte	0x40
	.long	0x9715
	.uleb128 0x23
	.byte	0x17
	.byte	0x8b
	.long	0x9709
	.uleb128 0x23
	.byte	0x17
	.byte	0x8d
	.long	0x972b
	.uleb128 0x23
	.byte	0x17
	.byte	0x8e
	.long	0x9741
	.uleb128 0x23
	.byte	0x17
	.byte	0x8f
	.long	0x975d
	.uleb128 0x23
	.byte	0x17
	.byte	0x90
	.long	0x978a
	.uleb128 0x23
	.byte	0x17
	.byte	0x91
	.long	0x97a5
	.uleb128 0x23
	.byte	0x17
	.byte	0x92
	.long	0x97cb
	.uleb128 0x23
	.byte	0x17
	.byte	0x93
	.long	0x97e6
	.uleb128 0x23
	.byte	0x17
	.byte	0x94
	.long	0x9802
	.uleb128 0x23
	.byte	0x17
	.byte	0x95
	.long	0x981e
	.uleb128 0x23
	.byte	0x17
	.byte	0x96
	.long	0x9834
	.uleb128 0x23
	.byte	0x17
	.byte	0x97
	.long	0x9840
	.uleb128 0x23
	.byte	0x17
	.byte	0x98
	.long	0x9866
	.uleb128 0x23
	.byte	0x17
	.byte	0x99
	.long	0x988b
	.uleb128 0x23
	.byte	0x17
	.byte	0x9a
	.long	0x98ac
	.uleb128 0x23
	.byte	0x17
	.byte	0x9b
	.long	0x98d7
	.uleb128 0x23
	.byte	0x17
	.byte	0x9c
	.long	0x98f2
	.uleb128 0x23
	.byte	0x17
	.byte	0x9e
	.long	0x9908
	.uleb128 0x23
	.byte	0x17
	.byte	0xa0
	.long	0x9929
	.uleb128 0x23
	.byte	0x17
	.byte	0xa1
	.long	0x9945
	.uleb128 0x23
	.byte	0x17
	.byte	0xa2
	.long	0x9960
	.uleb128 0x23
	.byte	0x17
	.byte	0xa4
	.long	0x9986
	.uleb128 0x23
	.byte	0x17
	.byte	0xa7
	.long	0x99a6
	.uleb128 0x23
	.byte	0x17
	.byte	0xaa
	.long	0x99cb
	.uleb128 0x23
	.byte	0x17
	.byte	0xac
	.long	0x99eb
	.uleb128 0x23
	.byte	0x17
	.byte	0xae
	.long	0x9a06
	.uleb128 0x23
	.byte	0x17
	.byte	0xb0
	.long	0x9a21
	.uleb128 0x23
	.byte	0x17
	.byte	0xb1
	.long	0x9a41
	.uleb128 0x23
	.byte	0x17
	.byte	0xb2
	.long	0x9a5b
	.uleb128 0x23
	.byte	0x17
	.byte	0xb3
	.long	0x9a75
	.uleb128 0x23
	.byte	0x17
	.byte	0xb4
	.long	0x9a8f
	.uleb128 0x23
	.byte	0x17
	.byte	0xb5
	.long	0x9aa9
	.uleb128 0x23
	.byte	0x17
	.byte	0xb6
	.long	0x9ac3
	.uleb128 0x23
	.byte	0x17
	.byte	0xb7
	.long	0x9b83
	.uleb128 0x23
	.byte	0x17
	.byte	0xb8
	.long	0x9b99
	.uleb128 0x23
	.byte	0x17
	.byte	0xb9
	.long	0x9bb9
	.uleb128 0x23
	.byte	0x17
	.byte	0xba
	.long	0x9bd8
	.uleb128 0x23
	.byte	0x17
	.byte	0xbb
	.long	0x9bf7
	.uleb128 0x23
	.byte	0x17
	.byte	0xbc
	.long	0x9c22
	.uleb128 0x23
	.byte	0x17
	.byte	0xbd
	.long	0x9c3d
	.uleb128 0x23
	.byte	0x17
	.byte	0xbf
	.long	0x9c5e
	.uleb128 0x23
	.byte	0x17
	.byte	0xc1
	.long	0x9c80
	.uleb128 0x23
	.byte	0x17
	.byte	0xc2
	.long	0x9ca0
	.uleb128 0x23
	.byte	0x17
	.byte	0xc3
	.long	0x9cc0
	.uleb128 0x23
	.byte	0x17
	.byte	0xc4
	.long	0x9ce0
	.uleb128 0x23
	.byte	0x17
	.byte	0xc5
	.long	0x9cff
	.uleb128 0x23
	.byte	0x17
	.byte	0xc6
	.long	0x9d15
	.uleb128 0x23
	.byte	0x17
	.byte	0xc7
	.long	0x9d35
	.uleb128 0x23
	.byte	0x17
	.byte	0xc8
	.long	0x9d54
	.uleb128 0x23
	.byte	0x17
	.byte	0xc9
	.long	0x9d73
	.uleb128 0x23
	.byte	0x17
	.byte	0xca
	.long	0x9d92
	.uleb128 0x23
	.byte	0x17
	.byte	0xcb
	.long	0x9da9
	.uleb128 0x23
	.byte	0x17
	.byte	0xcc
	.long	0x9dc0
	.uleb128 0x23
	.byte	0x17
	.byte	0xcd
	.long	0x9dde
	.uleb128 0x23
	.byte	0x17
	.byte	0xce
	.long	0x9dfd
	.uleb128 0x23
	.byte	0x17
	.byte	0xcf
	.long	0x9e1b
	.uleb128 0x23
	.byte	0x17
	.byte	0xd0
	.long	0x9e3a
	.uleb128 0x24
	.byte	0x17
	.value	0x108
	.long	0x9e5e
	.uleb128 0x24
	.byte	0x17
	.value	0x109
	.long	0x9e80
	.uleb128 0x24
	.byte	0x17
	.value	0x10a
	.long	0x9ea7
	.uleb128 0x24
	.byte	0x17
	.value	0x118
	.long	0x9c5e
	.uleb128 0x24
	.byte	0x17
	.value	0x11b
	.long	0x9986
	.uleb128 0x24
	.byte	0x17
	.value	0x11e
	.long	0x99cb
	.uleb128 0x24
	.byte	0x17
	.value	0x121
	.long	0x9a06
	.uleb128 0x24
	.byte	0x17
	.value	0x125
	.long	0x9e5e
	.uleb128 0x24
	.byte	0x17
	.value	0x126
	.long	0x9e80
	.uleb128 0x24
	.byte	0x17
	.value	0x127
	.long	0x9ea7
	.uleb128 0x5
	.long	.LASF263
	.byte	0x19
	.byte	0x36
	.long	0x1eb1
	.uleb128 0x6
	.long	.LASF265
	.byte	0x8
	.byte	0x19
	.byte	0x4b
	.long	0x1eab
	.uleb128 0x9
	.long	.LASF266
	.byte	0x19
	.byte	0x4d
	.long	0x919a
	.byte	0
	.uleb128 0x25
	.long	.LASF265
	.byte	0x19
	.byte	0x4f
	.long	.LASF267
	.long	0x1d1d
	.long	0x1d28
	.uleb128 0xb
	.long	0x9ece
	.uleb128 0xc
	.long	0x919a
	.byte	0
	.uleb128 0xa
	.long	.LASF268
	.byte	0x19
	.byte	0x51
	.long	.LASF269
	.long	0x1d3b
	.long	0x1d41
	.uleb128 0xb
	.long	0x9ece
	.byte	0
	.uleb128 0xa
	.long	.LASF270
	.byte	0x19
	.byte	0x52
	.long	.LASF271
	.long	0x1d54
	.long	0x1d5a
	.uleb128 0xb
	.long	0x9ece
	.byte	0
	.uleb128 0x17
	.long	.LASF272
	.byte	0x19
	.byte	0x54
	.long	.LASF273
	.long	0x919a
	.long	0x1d71
	.long	0x1d77
	.uleb128 0xb
	.long	0x9ed4
	.byte	0
	.uleb128 0x26
	.long	.LASF265
	.byte	0x19
	.byte	0x5a
	.long	.LASF274
	.byte	0x1
	.long	0x1d8b
	.long	0x1d91
	.uleb128 0xb
	.long	0x9ece
	.byte	0
	.uleb128 0x26
	.long	.LASF265
	.byte	0x19
	.byte	0x5c
	.long	.LASF275
	.byte	0x1
	.long	0x1da5
	.long	0x1db0
	.uleb128 0xb
	.long	0x9ece
	.uleb128 0xc
	.long	0x9eda
	.byte	0
	.uleb128 0x26
	.long	.LASF265
	.byte	0x19
	.byte	0x5f
	.long	.LASF276
	.byte	0x1
	.long	0x1dc4
	.long	0x1dcf
	.uleb128 0xb
	.long	0x9ece
	.uleb128 0xc
	.long	0x1eb8
	.byte	0
	.uleb128 0x26
	.long	.LASF265
	.byte	0x19
	.byte	0x63
	.long	.LASF277
	.byte	0x1
	.long	0x1de3
	.long	0x1dee
	.uleb128 0xb
	.long	0x9ece
	.uleb128 0xc
	.long	0x9ee5
	.byte	0
	.uleb128 0x27
	.long	.LASF89
	.byte	0x19
	.byte	0x70
	.long	.LASF278
	.long	0x9eeb
	.byte	0x1
	.long	0x1e06
	.long	0x1e11
	.uleb128 0xb
	.long	0x9ece
	.uleb128 0xc
	.long	0x9eda
	.byte	0
	.uleb128 0x27
	.long	.LASF89
	.byte	0x19
	.byte	0x74
	.long	.LASF279
	.long	0x9eeb
	.byte	0x1
	.long	0x1e29
	.long	0x1e34
	.uleb128 0xb
	.long	0x9ece
	.uleb128 0xc
	.long	0x9ee5
	.byte	0
	.uleb128 0x26
	.long	.LASF280
	.byte	0x19
	.byte	0x7b
	.long	.LASF281
	.byte	0x1
	.long	0x1e48
	.long	0x1e53
	.uleb128 0xb
	.long	0x9ece
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x26
	.long	.LASF205
	.byte	0x19
	.byte	0x7e
	.long	.LASF282
	.byte	0x1
	.long	0x1e67
	.long	0x1e72
	.uleb128 0xb
	.long	0x9ece
	.uleb128 0xc
	.long	0x9eeb
	.byte	0
	.uleb128 0x28
	.long	.LASF2016
	.byte	0x19
	.byte	0x8a
	.long	.LASF3071
	.long	0x9ef1
	.byte	0x1
	.long	0x1e8a
	.long	0x1e90
	.uleb128 0xb
	.long	0x9ed4
	.byte	0
	.uleb128 0x29
	.long	.LASF283
	.byte	0x19
	.byte	0x93
	.long	.LASF284
	.long	0x9ef8
	.byte	0x1
	.long	0x1ea4
	.uleb128 0xb
	.long	0x9ed4
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0x1cf2
	.byte	0
	.uleb128 0x23
	.byte	0x19
	.byte	0x3a
	.long	0x1cf2
	.uleb128 0x16
	.long	.LASF285
	.byte	0x18
	.byte	0xc8
	.long	0x9ee0
	.uleb128 0x2a
	.long	.LASF345
	.uleb128 0x14
	.long	0x1ec3
	.uleb128 0x2b
	.long	.LASF297
	.byte	0x1
	.byte	0x1a
	.byte	0x53
	.uleb128 0x7
	.long	.LASF286
	.byte	0x1
	.byte	0x1a
	.byte	0x88
	.long	0x1efe
	.uleb128 0xf
	.byte	0x4
	.long	0x913b
	.byte	0x1a
	.byte	0x8a
	.long	0x1ef4
	.uleb128 0x10
	.long	.LASF288
	.byte	0
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.byte	0
	.uleb128 0x7
	.long	.LASF289
	.byte	0x1
	.byte	0x1b
	.byte	0x45
	.long	0x1f51
	.uleb128 0x2d
	.long	.LASF294
	.byte	0x1b
	.byte	0x47
	.long	0x9efe
	.uleb128 0x16
	.long	.LASF290
	.byte	0x1b
	.byte	0x48
	.long	0x9ef1
	.uleb128 0x17
	.long	.LASF291
	.byte	0x1b
	.byte	0x4a
	.long	.LASF292
	.long	0x1f15
	.long	0x1f37
	.long	0x1f3d
	.uleb128 0xb
	.long	0x9f03
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x9ef1
	.uleb128 0x2e
	.string	"__v"
	.long	0x9ef1
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0x1efe
	.uleb128 0x7
	.long	.LASF293
	.byte	0x1
	.byte	0x1b
	.byte	0x45
	.long	0x1fa9
	.uleb128 0x2d
	.long	.LASF294
	.byte	0x1b
	.byte	0x47
	.long	0x9efe
	.uleb128 0x16
	.long	.LASF290
	.byte	0x1b
	.byte	0x48
	.long	0x9ef1
	.uleb128 0x17
	.long	.LASF295
	.byte	0x1b
	.byte	0x4a
	.long	.LASF296
	.long	0x1f6d
	.long	0x1f8f
	.long	0x1f95
	.uleb128 0xb
	.long	0x9f09
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x9ef1
	.uleb128 0x2e
	.string	"__v"
	.long	0x9ef1
	.byte	0x1
	.byte	0
	.uleb128 0x14
	.long	0x1f56
	.uleb128 0x2b
	.long	.LASF298
	.byte	0x1
	.byte	0x1c
	.byte	0x4c
	.uleb128 0x2b
	.long	.LASF299
	.byte	0x1
	.byte	0x1d
	.byte	0x59
	.uleb128 0x7
	.long	.LASF300
	.byte	0x1
	.byte	0x1d
	.byte	0x5f
	.long	0x1fd1
	.uleb128 0x8
	.long	0x1fb6
	.byte	0
	.byte	0
	.uleb128 0x7
	.long	.LASF301
	.byte	0x1
	.byte	0x1d
	.byte	0x63
	.long	0x1fe4
	.uleb128 0x8
	.long	0x1fbe
	.byte	0
	.byte	0
	.uleb128 0x7
	.long	.LASF302
	.byte	0x1
	.byte	0x1d
	.byte	0x67
	.long	0x1ff7
	.uleb128 0x8
	.long	0x1fd1
	.byte	0
	.byte	0
	.uleb128 0x2f
	.long	.LASF356
	.byte	0x2c
	.byte	0x30
	.uleb128 0x7
	.long	.LASF303
	.byte	0x1
	.byte	0xa
	.byte	0xe9
	.long	0x21c6
	.uleb128 0x16
	.long	.LASF304
	.byte	0xa
	.byte	0xeb
	.long	0x91a2
	.uleb128 0x16
	.long	.LASF305
	.byte	0xa
	.byte	0xec
	.long	0x30
	.uleb128 0x30
	.long	.LASF159
	.byte	0xa
	.byte	0xf2
	.long	.LASF444
	.long	0x203a
	.uleb128 0xc
	.long	0x9f27
	.uleb128 0xc
	.long	0x9f2d
	.byte	0
	.uleb128 0x14
	.long	0x200a
	.uleb128 0x31
	.string	"eq"
	.byte	0xa
	.byte	0xf6
	.long	.LASF306
	.long	0x9ef1
	.long	0x205c
	.uleb128 0xc
	.long	0x9f2d
	.uleb128 0xc
	.long	0x9f2d
	.byte	0
	.uleb128 0x31
	.string	"lt"
	.byte	0xa
	.byte	0xfa
	.long	.LASF307
	.long	0x9ef1
	.long	0x2079
	.uleb128 0xc
	.long	0x9f2d
	.uleb128 0xc
	.long	0x9f2d
	.byte	0
	.uleb128 0x1b
	.long	.LASF245
	.byte	0xa
	.value	0x102
	.long	.LASF308
	.long	0x30
	.long	0x209d
	.uleb128 0xc
	.long	0x9f33
	.uleb128 0xc
	.long	0x9f33
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0x1b
	.long	.LASF117
	.byte	0xa
	.value	0x10a
	.long	.LASF309
	.long	0x21c6
	.long	0x20b7
	.uleb128 0xc
	.long	0x9f33
	.byte	0
	.uleb128 0x1b
	.long	.LASF213
	.byte	0xa
	.value	0x10e
	.long	.LASF310
	.long	0x9f33
	.long	0x20db
	.uleb128 0xc
	.long	0x9f33
	.uleb128 0xc
	.long	0x21c6
	.uleb128 0xc
	.long	0x9f2d
	.byte	0
	.uleb128 0x1b
	.long	.LASF311
	.byte	0xa
	.value	0x116
	.long	.LASF312
	.long	0x9f39
	.long	0x20ff
	.uleb128 0xc
	.long	0x9f39
	.uleb128 0xc
	.long	0x9f33
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0x1b
	.long	.LASF203
	.byte	0xa
	.value	0x11e
	.long	.LASF313
	.long	0x9f39
	.long	0x2123
	.uleb128 0xc
	.long	0x9f39
	.uleb128 0xc
	.long	0x9f33
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0x1b
	.long	.LASF159
	.byte	0xa
	.value	0x126
	.long	.LASF314
	.long	0x9f39
	.long	0x2147
	.uleb128 0xc
	.long	0x9f39
	.uleb128 0xc
	.long	0x21c6
	.uleb128 0xc
	.long	0x200a
	.byte	0
	.uleb128 0x1b
	.long	.LASF315
	.byte	0xa
	.value	0x12e
	.long	.LASF316
	.long	0x200a
	.long	0x2161
	.uleb128 0xc
	.long	0x9f3f
	.byte	0
	.uleb128 0x14
	.long	0x2015
	.uleb128 0x1b
	.long	.LASF317
	.byte	0xa
	.value	0x134
	.long	.LASF318
	.long	0x2015
	.long	0x2180
	.uleb128 0xc
	.long	0x9f2d
	.byte	0
	.uleb128 0x1b
	.long	.LASF319
	.byte	0xa
	.value	0x138
	.long	.LASF320
	.long	0x9ef1
	.long	0x219f
	.uleb128 0xc
	.long	0x9f3f
	.uleb128 0xc
	.long	0x9f3f
	.byte	0
	.uleb128 0x32
	.string	"eof"
	.byte	0xa
	.value	0x13c
	.long	.LASF3072
	.long	0x2015
	.uleb128 0x33
	.long	.LASF321
	.byte	0xa
	.value	0x140
	.long	.LASF322
	.long	0x2015
	.uleb128 0xc
	.long	0x9f3f
	.byte	0
	.byte	0
	.uleb128 0x16
	.long	.LASF323
	.byte	0x18
	.byte	0xc4
	.long	0x9126
	.uleb128 0x23
	.byte	0x1e
	.byte	0x30
	.long	0x9f45
	.uleb128 0x23
	.byte	0x1e
	.byte	0x31
	.long	0x9f50
	.uleb128 0x23
	.byte	0x1e
	.byte	0x32
	.long	0x9f5b
	.uleb128 0x23
	.byte	0x1e
	.byte	0x33
	.long	0x9f66
	.uleb128 0x23
	.byte	0x1e
	.byte	0x35
	.long	0x9ff5
	.uleb128 0x23
	.byte	0x1e
	.byte	0x36
	.long	0xa000
	.uleb128 0x23
	.byte	0x1e
	.byte	0x37
	.long	0xa00b
	.uleb128 0x23
	.byte	0x1e
	.byte	0x38
	.long	0xa016
	.uleb128 0x23
	.byte	0x1e
	.byte	0x3a
	.long	0x9f9d
	.uleb128 0x23
	.byte	0x1e
	.byte	0x3b
	.long	0x9fa8
	.uleb128 0x23
	.byte	0x1e
	.byte	0x3c
	.long	0x9fb3
	.uleb128 0x23
	.byte	0x1e
	.byte	0x3d
	.long	0x9fbe
	.uleb128 0x23
	.byte	0x1e
	.byte	0x3f
	.long	0xa063
	.uleb128 0x23
	.byte	0x1e
	.byte	0x40
	.long	0xa04d
	.uleb128 0x23
	.byte	0x1e
	.byte	0x42
	.long	0x9f71
	.uleb128 0x23
	.byte	0x1e
	.byte	0x43
	.long	0x9f7c
	.uleb128 0x23
	.byte	0x1e
	.byte	0x44
	.long	0x9f87
	.uleb128 0x23
	.byte	0x1e
	.byte	0x45
	.long	0x9f92
	.uleb128 0x23
	.byte	0x1e
	.byte	0x47
	.long	0xa021
	.uleb128 0x23
	.byte	0x1e
	.byte	0x48
	.long	0xa02c
	.uleb128 0x23
	.byte	0x1e
	.byte	0x49
	.long	0xa037
	.uleb128 0x23
	.byte	0x1e
	.byte	0x4a
	.long	0xa042
	.uleb128 0x23
	.byte	0x1e
	.byte	0x4c
	.long	0x9fc9
	.uleb128 0x23
	.byte	0x1e
	.byte	0x4d
	.long	0x9fd4
	.uleb128 0x23
	.byte	0x1e
	.byte	0x4e
	.long	0x9fdf
	.uleb128 0x23
	.byte	0x1e
	.byte	0x4f
	.long	0x9fea
	.uleb128 0x23
	.byte	0x1e
	.byte	0x51
	.long	0xa06e
	.uleb128 0x23
	.byte	0x1e
	.byte	0x52
	.long	0xa058
	.uleb128 0x23
	.byte	0x1f
	.byte	0x35
	.long	0xa087
	.uleb128 0x23
	.byte	0x1f
	.byte	0x36
	.long	0xa1b4
	.uleb128 0x23
	.byte	0x1f
	.byte	0x37
	.long	0xa1ce
	.uleb128 0x2b
	.long	.LASF324
	.byte	0x1
	.byte	0x20
	.byte	0x52
	.uleb128 0x16
	.long	.LASF325
	.byte	0x18
	.byte	0xc5
	.long	0x915b
	.uleb128 0x16
	.long	.LASF326
	.byte	0x1b
	.byte	0x57
	.long	0x1f56
	.uleb128 0x6
	.long	.LASF327
	.byte	0x1
	.byte	0x21
	.byte	0x5c
	.long	0x2330
	.uleb128 0x34
	.long	0x79aa
	.byte	0
	.byte	0x1
	.uleb128 0x26
	.long	.LASF328
	.byte	0x21
	.byte	0x71
	.long	.LASF329
	.byte	0x1
	.long	0x22ef
	.long	0x22f5
	.uleb128 0xb
	.long	0xa21a
	.byte	0
	.uleb128 0x26
	.long	.LASF328
	.byte	0x21
	.byte	0x73
	.long	.LASF330
	.byte	0x1
	.long	0x2309
	.long	0x2314
	.uleb128 0xb
	.long	0xa21a
	.uleb128 0xc
	.long	0xa220
	.byte	0
	.uleb128 0x35
	.long	.LASF331
	.byte	0x21
	.byte	0x79
	.long	.LASF332
	.byte	0x1
	.long	0x2324
	.uleb128 0xb
	.long	0xa21a
	.uleb128 0xb
	.long	0x30
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0x22c8
	.uleb128 0x23
	.byte	0x22
	.byte	0x76
	.long	0xa24b
	.uleb128 0x23
	.byte	0x22
	.byte	0x77
	.long	0xa27b
	.uleb128 0x23
	.byte	0x22
	.byte	0x7b
	.long	0xa2dc
	.uleb128 0x23
	.byte	0x22
	.byte	0x7e
	.long	0xa2f9
	.uleb128 0x23
	.byte	0x22
	.byte	0x81
	.long	0xa313
	.uleb128 0x23
	.byte	0x22
	.byte	0x82
	.long	0xa32f
	.uleb128 0x23
	.byte	0x22
	.byte	0x83
	.long	0xa345
	.uleb128 0x23
	.byte	0x22
	.byte	0x84
	.long	0xa35b
	.uleb128 0x23
	.byte	0x22
	.byte	0x86
	.long	0xa384
	.uleb128 0x23
	.byte	0x22
	.byte	0x89
	.long	0xa39f
	.uleb128 0x23
	.byte	0x22
	.byte	0x8b
	.long	0xa3b5
	.uleb128 0x23
	.byte	0x22
	.byte	0x8e
	.long	0xa3d0
	.uleb128 0x23
	.byte	0x22
	.byte	0x8f
	.long	0xa3eb
	.uleb128 0x23
	.byte	0x22
	.byte	0x90
	.long	0xa40a
	.uleb128 0x23
	.byte	0x22
	.byte	0x92
	.long	0xa42a
	.uleb128 0x23
	.byte	0x22
	.byte	0x95
	.long	0xa44b
	.uleb128 0x23
	.byte	0x22
	.byte	0x98
	.long	0xa45d
	.uleb128 0x23
	.byte	0x22
	.byte	0x9a
	.long	0xa469
	.uleb128 0x23
	.byte	0x22
	.byte	0x9b
	.long	0xa47b
	.uleb128 0x23
	.byte	0x22
	.byte	0x9c
	.long	0xa49b
	.uleb128 0x23
	.byte	0x22
	.byte	0x9d
	.long	0xa4ba
	.uleb128 0x23
	.byte	0x22
	.byte	0x9e
	.long	0xa4d9
	.uleb128 0x23
	.byte	0x22
	.byte	0xa0
	.long	0xa4ef
	.uleb128 0x23
	.byte	0x22
	.byte	0xa1
	.long	0xa50e
	.uleb128 0x23
	.byte	0x22
	.byte	0xfe
	.long	0xa2ab
	.uleb128 0x24
	.byte	0x22
	.value	0x103
	.long	0x7b7c
	.uleb128 0x24
	.byte	0x22
	.value	0x104
	.long	0xa528
	.uleb128 0x24
	.byte	0x22
	.value	0x106
	.long	0xa543
	.uleb128 0x24
	.byte	0x22
	.value	0x107
	.long	0xa597
	.uleb128 0x24
	.byte	0x22
	.value	0x108
	.long	0xa559
	.uleb128 0x24
	.byte	0x22
	.value	0x109
	.long	0xa578
	.uleb128 0x24
	.byte	0x22
	.value	0x10a
	.long	0xa5b1
	.uleb128 0x36
	.long	.LASF333
	.byte	0x1
	.byte	0x23
	.value	0x1ba
	.long	0x2523
	.uleb128 0x37
	.long	.LASF9
	.byte	0x23
	.value	0x1bd
	.long	0x22c8
	.uleb128 0x37
	.long	.LASF290
	.byte	0x23
	.value	0x1bf
	.long	0x91a2
	.uleb128 0x37
	.long	.LASF4
	.byte	0x23
	.value	0x1c2
	.long	0x919c
	.uleb128 0x37
	.long	.LASF12
	.byte	0x23
	.value	0x1c5
	.long	0x9472
	.uleb128 0x37
	.long	.LASF334
	.byte	0x23
	.value	0x1cb
	.long	0xa1f5
	.uleb128 0x37
	.long	.LASF5
	.byte	0x23
	.value	0x1d1
	.long	0x21c6
	.uleb128 0x1b
	.long	.LASF335
	.byte	0x23
	.value	0x1ea
	.long	.LASF336
	.long	0x2441
	.long	0x2490
	.uleb128 0xc
	.long	0xa5cb
	.uleb128 0xc
	.long	0x2465
	.byte	0
	.uleb128 0x1b
	.long	.LASF335
	.byte	0x23
	.value	0x1f8
	.long	.LASF337
	.long	0x2441
	.long	0x24b4
	.uleb128 0xc
	.long	0xa5cb
	.uleb128 0xc
	.long	0x2465
	.uleb128 0xc
	.long	0x2459
	.byte	0
	.uleb128 0x1a
	.long	.LASF338
	.byte	0x23
	.value	0x204
	.long	.LASF339
	.long	0x24d4
	.uleb128 0xc
	.long	0xa5cb
	.uleb128 0xc
	.long	0x2441
	.uleb128 0xc
	.long	0x2465
	.byte	0
	.uleb128 0x1b
	.long	.LASF119
	.byte	0x23
	.value	0x226
	.long	.LASF340
	.long	0x2465
	.long	0x24ee
	.uleb128 0xc
	.long	0xa5d1
	.byte	0
	.uleb128 0x14
	.long	0x2429
	.uleb128 0x1b
	.long	.LASF341
	.byte	0x23
	.value	0x22f
	.long	.LASF342
	.long	0x2429
	.long	0x250d
	.uleb128 0xc
	.long	0xa5d1
	.byte	0
	.uleb128 0x37
	.long	.LASF343
	.byte	0x23
	.value	0x1dd
	.long	0x22c8
	.uleb128 0x20
	.long	.LASF261
	.long	0x22c8
	.byte	0
	.uleb128 0x16
	.long	.LASF344
	.byte	0x1b
	.byte	0x5a
	.long	0x1efe
	.uleb128 0x2a
	.long	.LASF346
	.uleb128 0x2a
	.long	.LASF347
	.uleb128 0x6
	.long	.LASF348
	.byte	0x10
	.byte	0x24
	.byte	0x2f
	.long	0x2620
	.uleb128 0xe
	.long	.LASF13
	.byte	0x24
	.byte	0x36
	.long	0x9472
	.byte	0x1
	.uleb128 0x9
	.long	.LASF349
	.byte	0x24
	.byte	0x3a
	.long	0x2544
	.byte	0
	.uleb128 0xe
	.long	.LASF5
	.byte	0x24
	.byte	0x35
	.long	0x21c6
	.byte	0x1
	.uleb128 0x9
	.long	.LASF350
	.byte	0x24
	.byte	0x3b
	.long	0x255c
	.byte	0x8
	.uleb128 0xe
	.long	.LASF14
	.byte	0x24
	.byte	0x37
	.long	0x9472
	.byte	0x1
	.uleb128 0xa
	.long	.LASF351
	.byte	0x24
	.byte	0x3e
	.long	.LASF352
	.long	0x2593
	.long	0x25a3
	.uleb128 0xb
	.long	0xa62f
	.uleb128 0xc
	.long	0x2574
	.uleb128 0xc
	.long	0x255c
	.byte	0
	.uleb128 0x26
	.long	.LASF351
	.byte	0x24
	.byte	0x42
	.long	.LASF353
	.byte	0x1
	.long	0x25b7
	.long	0x25bd
	.uleb128 0xb
	.long	0xa62f
	.byte	0
	.uleb128 0x27
	.long	.LASF115
	.byte	0x24
	.byte	0x47
	.long	.LASF354
	.long	0x255c
	.byte	0x1
	.long	0x25d5
	.long	0x25db
	.uleb128 0xb
	.long	0xa635
	.byte	0
	.uleb128 0x27
	.long	.LASF96
	.byte	0x24
	.byte	0x4b
	.long	.LASF355
	.long	0x2574
	.byte	0x1
	.long	0x25f3
	.long	0x25f9
	.uleb128 0xb
	.long	0xa635
	.byte	0
	.uleb128 0x38
	.string	"end"
	.byte	0x24
	.byte	0x4f
	.long	.LASF616
	.long	0x2574
	.byte	0x1
	.long	0x2611
	.long	0x2617
	.uleb128 0xb
	.long	0xa635
	.byte	0
	.uleb128 0x2c
	.string	"_E"
	.long	0x91a2
	.byte	0
	.uleb128 0x14
	.long	0x2538
	.uleb128 0x39
	.string	"_V2"
	.byte	0x25
	.byte	0x3f
	.uleb128 0x22
	.byte	0x25
	.byte	0x3f
	.long	0x2625
	.uleb128 0x3a
	.long	.LASF378
	.byte	0x4
	.long	0x30
	.byte	0x26
	.byte	0x39
	.long	0x26d4
	.uleb128 0x10
	.long	.LASF357
	.byte	0x1
	.uleb128 0x10
	.long	.LASF358
	.byte	0x2
	.uleb128 0x10
	.long	.LASF359
	.byte	0x4
	.uleb128 0x10
	.long	.LASF360
	.byte	0x8
	.uleb128 0x10
	.long	.LASF361
	.byte	0x10
	.uleb128 0x10
	.long	.LASF362
	.byte	0x20
	.uleb128 0x10
	.long	.LASF363
	.byte	0x40
	.uleb128 0x10
	.long	.LASF364
	.byte	0x80
	.uleb128 0x3b
	.long	.LASF365
	.value	0x100
	.uleb128 0x3b
	.long	.LASF366
	.value	0x200
	.uleb128 0x3b
	.long	.LASF367
	.value	0x400
	.uleb128 0x3b
	.long	.LASF368
	.value	0x800
	.uleb128 0x3b
	.long	.LASF369
	.value	0x1000
	.uleb128 0x3b
	.long	.LASF370
	.value	0x2000
	.uleb128 0x3b
	.long	.LASF371
	.value	0x4000
	.uleb128 0x10
	.long	.LASF372
	.byte	0xb0
	.uleb128 0x10
	.long	.LASF373
	.byte	0x4a
	.uleb128 0x3b
	.long	.LASF374
	.value	0x104
	.uleb128 0x3c
	.long	.LASF375
	.long	0x10000
	.uleb128 0x3c
	.long	.LASF376
	.long	0x7fffffff
	.uleb128 0x3d
	.long	.LASF377
	.sleb128 -2147483648
	.byte	0
	.uleb128 0x3a
	.long	.LASF379
	.byte	0x4
	.long	0x30
	.byte	0x26
	.byte	0x6f
	.long	0x2725
	.uleb128 0x10
	.long	.LASF380
	.byte	0x1
	.uleb128 0x10
	.long	.LASF381
	.byte	0x2
	.uleb128 0x10
	.long	.LASF382
	.byte	0x4
	.uleb128 0x10
	.long	.LASF383
	.byte	0x8
	.uleb128 0x10
	.long	.LASF384
	.byte	0x10
	.uleb128 0x10
	.long	.LASF385
	.byte	0x20
	.uleb128 0x3c
	.long	.LASF386
	.long	0x10000
	.uleb128 0x3c
	.long	.LASF387
	.long	0x7fffffff
	.uleb128 0x3d
	.long	.LASF388
	.sleb128 -2147483648
	.byte	0
	.uleb128 0x3a
	.long	.LASF389
	.byte	0x4
	.long	0x30
	.byte	0x26
	.byte	0x99
	.long	0x276a
	.uleb128 0x10
	.long	.LASF390
	.byte	0
	.uleb128 0x10
	.long	.LASF391
	.byte	0x1
	.uleb128 0x10
	.long	.LASF392
	.byte	0x2
	.uleb128 0x10
	.long	.LASF393
	.byte	0x4
	.uleb128 0x3c
	.long	.LASF394
	.long	0x10000
	.uleb128 0x3c
	.long	.LASF395
	.long	0x7fffffff
	.uleb128 0x3d
	.long	.LASF396
	.sleb128 -2147483648
	.byte	0
	.uleb128 0x3a
	.long	.LASF397
	.byte	0x4
	.long	0x913b
	.byte	0x26
	.byte	0xc1
	.long	0x2796
	.uleb128 0x10
	.long	.LASF398
	.byte	0
	.uleb128 0x10
	.long	.LASF399
	.byte	0x1
	.uleb128 0x10
	.long	.LASF400
	.byte	0x2
	.uleb128 0x3c
	.long	.LASF401
	.long	0x10000
	.byte	0
	.uleb128 0x3e
	.long	.LASF433
	.long	0x29ff
	.uleb128 0x3f
	.long	.LASF404
	.byte	0x1
	.byte	0x26
	.value	0x259
	.byte	0x1
	.long	0x27fd
	.uleb128 0x40
	.long	.LASF402
	.byte	0x26
	.value	0x261
	.long	0xa1ea
	.uleb128 0x40
	.long	.LASF403
	.byte	0x26
	.value	0x262
	.long	0x9ef1
	.uleb128 0x1c
	.long	.LASF404
	.byte	0x26
	.value	0x25d
	.long	.LASF405
	.byte	0x1
	.long	0x27da
	.long	0x27e0
	.uleb128 0xb
	.long	0xa646
	.byte	0
	.uleb128 0x41
	.long	.LASF406
	.byte	0x26
	.value	0x25e
	.long	.LASF407
	.byte	0x1
	.long	0x27f1
	.uleb128 0xb
	.long	0xa646
	.uleb128 0xb
	.long	0x30
	.byte	0
	.byte	0
	.uleb128 0x42
	.long	.LASF408
	.byte	0x26
	.value	0x18e
	.long	0x2725
	.byte	0x1
	.uleb128 0x42
	.long	.LASF409
	.byte	0x26
	.value	0x143
	.long	0x2633
	.byte	0x1
	.uleb128 0x43
	.long	.LASF410
	.byte	0x26
	.value	0x146
	.long	0x2825
	.byte	0x1
	.byte	0x1
	.uleb128 0x14
	.long	0x280a
	.uleb128 0x44
	.string	"dec"
	.byte	0x26
	.value	0x149
	.long	0x2825
	.byte	0x1
	.byte	0x2
	.uleb128 0x43
	.long	.LASF411
	.byte	0x26
	.value	0x14c
	.long	0x2825
	.byte	0x1
	.byte	0x4
	.uleb128 0x44
	.string	"hex"
	.byte	0x26
	.value	0x14f
	.long	0x2825
	.byte	0x1
	.byte	0x8
	.uleb128 0x43
	.long	.LASF412
	.byte	0x26
	.value	0x154
	.long	0x2825
	.byte	0x1
	.byte	0x10
	.uleb128 0x43
	.long	.LASF413
	.byte	0x26
	.value	0x158
	.long	0x2825
	.byte	0x1
	.byte	0x20
	.uleb128 0x44
	.string	"oct"
	.byte	0x26
	.value	0x15b
	.long	0x2825
	.byte	0x1
	.byte	0x40
	.uleb128 0x43
	.long	.LASF414
	.byte	0x26
	.value	0x15f
	.long	0x2825
	.byte	0x1
	.byte	0x80
	.uleb128 0x45
	.long	.LASF415
	.byte	0x26
	.value	0x162
	.long	0x2825
	.byte	0x1
	.value	0x100
	.uleb128 0x45
	.long	.LASF416
	.byte	0x26
	.value	0x166
	.long	0x2825
	.byte	0x1
	.value	0x200
	.uleb128 0x45
	.long	.LASF417
	.byte	0x26
	.value	0x16a
	.long	0x2825
	.byte	0x1
	.value	0x400
	.uleb128 0x45
	.long	.LASF418
	.byte	0x26
	.value	0x16d
	.long	0x2825
	.byte	0x1
	.value	0x800
	.uleb128 0x45
	.long	.LASF419
	.byte	0x26
	.value	0x170
	.long	0x2825
	.byte	0x1
	.value	0x1000
	.uleb128 0x45
	.long	.LASF420
	.byte	0x26
	.value	0x173
	.long	0x2825
	.byte	0x1
	.value	0x2000
	.uleb128 0x45
	.long	.LASF421
	.byte	0x26
	.value	0x177
	.long	0x2825
	.byte	0x1
	.value	0x4000
	.uleb128 0x43
	.long	.LASF422
	.byte	0x26
	.value	0x17a
	.long	0x2825
	.byte	0x1
	.byte	0xb0
	.uleb128 0x43
	.long	.LASF423
	.byte	0x26
	.value	0x17d
	.long	0x2825
	.byte	0x1
	.byte	0x4a
	.uleb128 0x45
	.long	.LASF424
	.byte	0x26
	.value	0x180
	.long	0x2825
	.byte	0x1
	.value	0x104
	.uleb128 0x43
	.long	.LASF425
	.byte	0x26
	.value	0x192
	.long	0x292e
	.byte	0x1
	.byte	0x1
	.uleb128 0x14
	.long	0x27fd
	.uleb128 0x43
	.long	.LASF426
	.byte	0x26
	.value	0x195
	.long	0x292e
	.byte	0x1
	.byte	0x2
	.uleb128 0x43
	.long	.LASF427
	.byte	0x26
	.value	0x19a
	.long	0x292e
	.byte	0x1
	.byte	0x4
	.uleb128 0x43
	.long	.LASF428
	.byte	0x26
	.value	0x19d
	.long	0x292e
	.byte	0x1
	.byte	0
	.uleb128 0x42
	.long	.LASF429
	.byte	0x26
	.value	0x1ad
	.long	0x26d4
	.byte	0x1
	.uleb128 0x44
	.string	"app"
	.byte	0x26
	.value	0x1b0
	.long	0x2978
	.byte	0x1
	.byte	0x1
	.uleb128 0x14
	.long	0x295d
	.uleb128 0x44
	.string	"ate"
	.byte	0x26
	.value	0x1b3
	.long	0x2978
	.byte	0x1
	.byte	0x2
	.uleb128 0x43
	.long	.LASF430
	.byte	0x26
	.value	0x1b8
	.long	0x2978
	.byte	0x1
	.byte	0x4
	.uleb128 0x44
	.string	"in"
	.byte	0x26
	.value	0x1bb
	.long	0x2978
	.byte	0x1
	.byte	0x8
	.uleb128 0x44
	.string	"out"
	.byte	0x26
	.value	0x1be
	.long	0x2978
	.byte	0x1
	.byte	0x10
	.uleb128 0x43
	.long	.LASF431
	.byte	0x26
	.value	0x1c1
	.long	0x2978
	.byte	0x1
	.byte	0x20
	.uleb128 0x42
	.long	.LASF432
	.byte	0x26
	.value	0x1cd
	.long	0x276a
	.byte	0x1
	.uleb128 0x44
	.string	"beg"
	.byte	0x26
	.value	0x1d0
	.long	0x29dd
	.byte	0x1
	.byte	0
	.uleb128 0x14
	.long	0x29c2
	.uleb128 0x44
	.string	"cur"
	.byte	0x26
	.value	0x1d3
	.long	0x29dd
	.byte	0x1
	.byte	0x1
	.uleb128 0x44
	.string	"end"
	.byte	0x26
	.value	0x1d6
	.long	0x29dd
	.byte	0x1
	.byte	0x2
	.byte	0
	.uleb128 0x23
	.byte	0x27
	.byte	0x52
	.long	0xa657
	.uleb128 0x23
	.byte	0x27
	.byte	0x53
	.long	0xa64c
	.uleb128 0x23
	.byte	0x27
	.byte	0x54
	.long	0x9709
	.uleb128 0x23
	.byte	0x27
	.byte	0x5c
	.long	0xa66d
	.uleb128 0x23
	.byte	0x27
	.byte	0x65
	.long	0xa687
	.uleb128 0x23
	.byte	0x27
	.byte	0x68
	.long	0xa6a1
	.uleb128 0x23
	.byte	0x27
	.byte	0x69
	.long	0xa6b6
	.uleb128 0x3e
	.long	.LASF434
	.long	0x2abc
	.uleb128 0x27
	.long	.LASF435
	.byte	0xc
	.byte	0xdc
	.long	.LASF436
	.long	0x1512f
	.byte	0x1
	.long	0x2a51
	.long	0x2a5c
	.uleb128 0xb
	.long	0x15135
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0xe
	.long	.LASF437
	.byte	0xc
	.byte	0x47
	.long	0x2a30
	.byte	0x1
	.uleb128 0x27
	.long	.LASF435
	.byte	0xc
	.byte	0xe0
	.long	.LASF438
	.long	0x1512f
	.byte	0x1
	.long	0x2a80
	.long	0x2a8b
	.uleb128 0xb
	.long	0x15135
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x20
	.long	.LASF259
	.long	0x91a2
	.uleb128 0x21
	.long	.LASF260
	.long	0x1ffe
	.uleb128 0x46
	.long	.LASF439
	.long	.LASF441
	.byte	0x28
	.byte	0x3f
	.long	.LASF439
	.uleb128 0x46
	.long	.LASF440
	.long	.LASF435
	.byte	0x28
	.byte	0x69
	.long	.LASF440
	.byte	0
	.uleb128 0x23
	.byte	0x29
	.byte	0x4b
	.long	0xa6db
	.uleb128 0x23
	.byte	0x29
	.byte	0x52
	.long	0xa6fe
	.uleb128 0x23
	.byte	0x29
	.byte	0x55
	.long	0xa718
	.uleb128 0x23
	.byte	0x29
	.byte	0x5b
	.long	0xa72e
	.uleb128 0x23
	.byte	0x29
	.byte	0x5c
	.long	0xa749
	.uleb128 0x23
	.byte	0x29
	.byte	0x5d
	.long	0xa768
	.uleb128 0x23
	.byte	0x29
	.byte	0x5e
	.long	0xa786
	.uleb128 0x23
	.byte	0x29
	.byte	0x5f
	.long	0xa7a5
	.uleb128 0x23
	.byte	0x29
	.byte	0x60
	.long	0xa7c3
	.uleb128 0x47
	.byte	0x2a
	.value	0x216
	.long	0x2625
	.uleb128 0x7
	.long	.LASF442
	.byte	0x1
	.byte	0x10
	.byte	0x6c
	.long	0x2b52
	.uleb128 0x30
	.long	.LASF443
	.byte	0x10
	.byte	0x70
	.long	.LASF445
	.long	0x2b32
	.uleb128 0x20
	.long	.LASF446
	.long	0xaa8a
	.uleb128 0xc
	.long	0xaa8a
	.uleb128 0xc
	.long	0xaa8a
	.byte	0
	.uleb128 0x48
	.long	.LASF448
	.byte	0x10
	.byte	0x70
	.long	.LASF1022
	.uleb128 0x20
	.long	.LASF446
	.long	0x14128
	.uleb128 0xc
	.long	0x14128
	.uleb128 0xc
	.long	0x14128
	.byte	0
	.byte	0
	.uleb128 0x24
	.byte	0x2b
	.value	0x42b
	.long	0xa7ed
	.uleb128 0x24
	.byte	0x2b
	.value	0x42c
	.long	0xa7e2
	.uleb128 0x36
	.long	.LASF449
	.byte	0x1
	.byte	0x12
	.value	0x213
	.long	0x2b9d
	.uleb128 0x33
	.long	.LASF450
	.byte	0x12
	.value	0x217
	.long	.LASF451
	.long	0x14128
	.uleb128 0x20
	.long	.LASF446
	.long	0x14128
	.uleb128 0x20
	.long	.LASF452
	.long	0x9126
	.uleb128 0xc
	.long	0x14128
	.uleb128 0xc
	.long	0x9126
	.byte	0
	.byte	0
	.uleb128 0x2f
	.long	.LASF453
	.byte	0x2d
	.byte	0x2a
	.uleb128 0x49
	.long	.LASF454
	.value	0x1388
	.byte	0x2e
	.value	0x1bc
	.long	0x2def
	.uleb128 0x4a
	.long	.LASF456
	.byte	0x2e
	.value	0x1de
	.long	0x2def
	.byte	0x1
	.uleb128 0x45
	.long	.LASF457
	.byte	0x2e
	.value	0x1df
	.long	0x2def
	.byte	0x1
	.value	0x270
	.uleb128 0x4a
	.long	.LASF458
	.byte	0x2e
	.value	0x1e0
	.long	0x2def
	.byte	0x1
	.uleb128 0x4a
	.long	.LASF459
	.byte	0x2e
	.value	0x1e1
	.long	0x2def
	.byte	0x1
	.uleb128 0x42
	.long	.LASF460
	.byte	0x2e
	.value	0x1db
	.long	0x9126
	.byte	0x1
	.uleb128 0x4a
	.long	.LASF461
	.byte	0x2e
	.value	0x1e2
	.long	0x2c02
	.byte	0x1
	.uleb128 0x14
	.long	0x2be8
	.uleb128 0x4a
	.long	.LASF462
	.byte	0x2e
	.value	0x1e3
	.long	0x2def
	.byte	0x1
	.uleb128 0x4a
	.long	.LASF463
	.byte	0x2e
	.value	0x1e4
	.long	0x2c02
	.byte	0x1
	.uleb128 0x4a
	.long	.LASF464
	.byte	0x2e
	.value	0x1e5
	.long	0x2def
	.byte	0x1
	.uleb128 0x4a
	.long	.LASF465
	.byte	0x2e
	.value	0x1e6
	.long	0x2c02
	.byte	0x1
	.uleb128 0x4a
	.long	.LASF466
	.byte	0x2e
	.value	0x1e7
	.long	0x2def
	.byte	0x1
	.uleb128 0x4a
	.long	.LASF467
	.byte	0x2e
	.value	0x1e8
	.long	0x2c02
	.byte	0x1
	.uleb128 0x4a
	.long	.LASF468
	.byte	0x2e
	.value	0x1e9
	.long	0x2def
	.byte	0x1
	.uleb128 0x4a
	.long	.LASF469
	.byte	0x2e
	.value	0x1ea
	.long	0x2c02
	.byte	0x1
	.uleb128 0x4a
	.long	.LASF470
	.byte	0x2e
	.value	0x1eb
	.long	0x2c02
	.byte	0x1
	.uleb128 0x4b
	.long	.LASF471
	.byte	0x2e
	.value	0x266
	.long	0xb2e7
	.byte	0
	.uleb128 0x4c
	.long	.LASF6
	.byte	0x2e
	.value	0x267
	.long	0x21c6
	.value	0x1380
	.uleb128 0x1d
	.long	.LASF472
	.byte	0x2e
	.value	0x1ef
	.long	.LASF473
	.byte	0x1
	.long	0x2cac
	.long	0x2cb7
	.uleb128 0xb
	.long	0xb2f9
	.uleb128 0xc
	.long	0x2be8
	.byte	0
	.uleb128 0x1c
	.long	.LASF474
	.byte	0x2e
	.value	0x200
	.long	.LASF475
	.byte	0x1
	.long	0x2ccc
	.long	0x2cd7
	.uleb128 0xb
	.long	0xb2f9
	.uleb128 0xc
	.long	0x2be8
	.byte	0
	.uleb128 0x4d
	.string	"min"
	.byte	0x2e
	.value	0x20a
	.long	.LASF476
	.long	0x2be8
	.byte	0x1
	.uleb128 0x4d
	.string	"max"
	.byte	0x2e
	.value	0x211
	.long	.LASF477
	.long	0x2be8
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF478
	.byte	0x2e
	.value	0x218
	.long	.LASF479
	.byte	0x1
	.long	0x2d0e
	.long	0x2d19
	.uleb128 0xb
	.long	0xb2f9
	.uleb128 0xc
	.long	0x9ec7
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x2e
	.value	0x21b
	.long	.LASF481
	.long	0x2be8
	.byte	0x1
	.long	0x2d32
	.long	0x2d38
	.uleb128 0xb
	.long	0xb2f9
	.byte	0
	.uleb128 0x19
	.long	.LASF482
	.byte	0x2e
	.value	0x264
	.long	.LASF483
	.long	0x2d4c
	.long	0x2d52
	.uleb128 0xb
	.long	0xb2f9
	.byte	0
	.uleb128 0x20
	.long	.LASF484
	.long	0x9126
	.uleb128 0x2e
	.string	"__w"
	.long	0x9126
	.byte	0x20
	.uleb128 0x4e
	.string	"__n"
	.long	0x9126
	.value	0x270
	.uleb128 0x4e
	.string	"__m"
	.long	0x9126
	.value	0x18d
	.uleb128 0x2e
	.string	"__r"
	.long	0x9126
	.byte	0x1f
	.uleb128 0x4f
	.string	"__a"
	.long	0x9126
	.long	0x9908b0df
	.uleb128 0x2e
	.string	"__u"
	.long	0x9126
	.byte	0xb
	.uleb128 0x4f
	.string	"__d"
	.long	0x9126
	.long	0xffffffff
	.uleb128 0x2e
	.string	"__s"
	.long	0x9126
	.byte	0x7
	.uleb128 0x4f
	.string	"__b"
	.long	0x9126
	.long	0x9d2c5680
	.uleb128 0x2e
	.string	"__t"
	.long	0x9126
	.byte	0xf
	.uleb128 0x4f
	.string	"__c"
	.long	0x9126
	.long	0xefc60000
	.uleb128 0x2e
	.string	"__l"
	.long	0x9126
	.byte	0x12
	.uleb128 0x4f
	.string	"__f"
	.long	0x9126
	.long	0x6c078965
	.byte	0
	.uleb128 0x14
	.long	0x21c6
	.uleb128 0x36
	.long	.LASF485
	.byte	0x1
	.byte	0x23
	.value	0x1ba
	.long	0x2eef
	.uleb128 0x37
	.long	.LASF9
	.byte	0x23
	.value	0x1bd
	.long	0x2eef
	.uleb128 0x37
	.long	.LASF290
	.byte	0x23
	.value	0x1bf
	.long	0x29
	.uleb128 0x37
	.long	.LASF4
	.byte	0x23
	.value	0x1c2
	.long	0xab81
	.uleb128 0x37
	.long	.LASF334
	.byte	0x23
	.value	0x1cb
	.long	0xa1f5
	.uleb128 0x37
	.long	.LASF5
	.byte	0x23
	.value	0x1d1
	.long	0x21c6
	.uleb128 0x1b
	.long	.LASF335
	.byte	0x23
	.value	0x1ea
	.long	.LASF486
	.long	0x2e19
	.long	0x2e5c
	.uleb128 0xc
	.long	0xb30a
	.uleb128 0xc
	.long	0x2e31
	.byte	0
	.uleb128 0x1b
	.long	.LASF335
	.byte	0x23
	.value	0x1f8
	.long	.LASF487
	.long	0x2e19
	.long	0x2e80
	.uleb128 0xc
	.long	0xb30a
	.uleb128 0xc
	.long	0x2e31
	.uleb128 0xc
	.long	0x2e25
	.byte	0
	.uleb128 0x1a
	.long	.LASF338
	.byte	0x23
	.value	0x204
	.long	.LASF488
	.long	0x2ea0
	.uleb128 0xc
	.long	0xb30a
	.uleb128 0xc
	.long	0x2e19
	.uleb128 0xc
	.long	0x2e31
	.byte	0
	.uleb128 0x1b
	.long	.LASF119
	.byte	0x23
	.value	0x226
	.long	.LASF489
	.long	0x2e31
	.long	0x2eba
	.uleb128 0xc
	.long	0xb310
	.byte	0
	.uleb128 0x14
	.long	0x2e01
	.uleb128 0x1b
	.long	.LASF341
	.byte	0x23
	.value	0x22f
	.long	.LASF490
	.long	0x2e01
	.long	0x2ed9
	.uleb128 0xc
	.long	0xb310
	.byte	0
	.uleb128 0x37
	.long	.LASF343
	.byte	0x23
	.value	0x1dd
	.long	0x2eef
	.uleb128 0x20
	.long	.LASF261
	.long	0x2eef
	.byte	0
	.uleb128 0x6
	.long	.LASF491
	.byte	0x1
	.byte	0x21
	.byte	0x5c
	.long	0x2f57
	.uleb128 0x34
	.long	0x83ec
	.byte	0
	.byte	0x1
	.uleb128 0x26
	.long	.LASF328
	.byte	0x21
	.byte	0x71
	.long	.LASF492
	.byte	0x1
	.long	0x2f16
	.long	0x2f1c
	.uleb128 0xb
	.long	0xb34c
	.byte	0
	.uleb128 0x26
	.long	.LASF328
	.byte	0x21
	.byte	0x73
	.long	.LASF493
	.byte	0x1
	.long	0x2f30
	.long	0x2f3b
	.uleb128 0xb
	.long	0xb34c
	.uleb128 0xc
	.long	0xb322
	.byte	0
	.uleb128 0x35
	.long	.LASF331
	.byte	0x21
	.byte	0x79
	.long	.LASF494
	.byte	0x1
	.long	0x2f4b
	.uleb128 0xb
	.long	0xb34c
	.uleb128 0xb
	.long	0x30
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0x2eef
	.uleb128 0x7
	.long	.LASF495
	.byte	0x18
	.byte	0xf
	.byte	0x48
	.long	0x3208
	.uleb128 0x7
	.long	.LASF496
	.byte	0x18
	.byte	0xf
	.byte	0x4f
	.long	0x300e
	.uleb128 0x8
	.long	0x2eef
	.byte	0
	.uleb128 0x9
	.long	.LASF497
	.byte	0xf
	.byte	0x52
	.long	0x300e
	.byte	0
	.uleb128 0x9
	.long	.LASF498
	.byte	0xf
	.byte	0x53
	.long	0x300e
	.byte	0x8
	.uleb128 0x9
	.long	.LASF499
	.byte	0xf
	.byte	0x54
	.long	0x300e
	.byte	0x10
	.uleb128 0xa
	.long	.LASF496
	.byte	0xf
	.byte	0x56
	.long	.LASF500
	.long	0x2fb1
	.long	0x2fb7
	.uleb128 0xb
	.long	0xb352
	.byte	0
	.uleb128 0xa
	.long	.LASF496
	.byte	0xf
	.byte	0x5a
	.long	.LASF501
	.long	0x2fca
	.long	0x2fd5
	.uleb128 0xb
	.long	0xb352
	.uleb128 0xc
	.long	0xb358
	.byte	0
	.uleb128 0xa
	.long	.LASF496
	.byte	0xf
	.byte	0x5f
	.long	.LASF502
	.long	0x2fe8
	.long	0x2ff3
	.uleb128 0xb
	.long	0xb352
	.uleb128 0xc
	.long	0xb35e
	.byte	0
	.uleb128 0x50
	.long	.LASF503
	.byte	0xf
	.byte	0x65
	.long	.LASF504
	.long	0x3002
	.uleb128 0xb
	.long	0xb352
	.uleb128 0xc
	.long	0xb364
	.byte	0
	.byte	0
	.uleb128 0x16
	.long	.LASF4
	.byte	0xf
	.byte	0x4d
	.long	0x830e
	.uleb128 0x16
	.long	.LASF505
	.byte	0xf
	.byte	0x4b
	.long	0x83cd
	.uleb128 0x14
	.long	0x3019
	.uleb128 0x9
	.long	.LASF506
	.byte	0xf
	.byte	0xa4
	.long	0x2f68
	.byte	0
	.uleb128 0x16
	.long	.LASF9
	.byte	0xf
	.byte	0x6e
	.long	0x2eef
	.uleb128 0x17
	.long	.LASF507
	.byte	0xf
	.byte	0x71
	.long	.LASF508
	.long	0xb36a
	.long	0x3057
	.long	0x305d
	.uleb128 0xb
	.long	0xb370
	.byte	0
	.uleb128 0x17
	.long	.LASF507
	.byte	0xf
	.byte	0x75
	.long	.LASF509
	.long	0xb358
	.long	0x3074
	.long	0x307a
	.uleb128 0xb
	.long	0xb376
	.byte	0
	.uleb128 0x17
	.long	.LASF211
	.byte	0xf
	.byte	0x79
	.long	.LASF510
	.long	0x3035
	.long	0x3091
	.long	0x3097
	.uleb128 0xb
	.long	0xb376
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x7c
	.long	.LASF512
	.long	0x30aa
	.long	0x30b0
	.uleb128 0xb
	.long	0xb370
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x7f
	.long	.LASF513
	.long	0x30c3
	.long	0x30ce
	.uleb128 0xb
	.long	0xb370
	.uleb128 0xc
	.long	0xb37c
	.byte	0
	.uleb128 0x14
	.long	0x3035
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x82
	.long	.LASF514
	.long	0x30e6
	.long	0x30f1
	.uleb128 0xb
	.long	0xb370
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x86
	.long	.LASF515
	.long	0x3104
	.long	0x3114
	.uleb128 0xb
	.long	0xb370
	.uleb128 0xc
	.long	0x21c6
	.uleb128 0xc
	.long	0xb37c
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x8b
	.long	.LASF516
	.long	0x3127
	.long	0x3132
	.uleb128 0xb
	.long	0xb370
	.uleb128 0xc
	.long	0xb35e
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x8e
	.long	.LASF517
	.long	0x3145
	.long	0x3150
	.uleb128 0xb
	.long	0xb370
	.uleb128 0xc
	.long	0xb382
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x92
	.long	.LASF518
	.long	0x3163
	.long	0x3173
	.uleb128 0xb
	.long	0xb370
	.uleb128 0xc
	.long	0xb382
	.uleb128 0xc
	.long	0xb37c
	.byte	0
	.uleb128 0xa
	.long	.LASF519
	.byte	0xf
	.byte	0x9f
	.long	.LASF520
	.long	0x3186
	.long	0x3191
	.uleb128 0xb
	.long	0xb370
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x17
	.long	.LASF521
	.byte	0xf
	.byte	0xa7
	.long	.LASF522
	.long	0x300e
	.long	0x31a8
	.long	0x31b3
	.uleb128 0xb
	.long	0xb370
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0xa
	.long	.LASF523
	.byte	0xf
	.byte	0xae
	.long	.LASF524
	.long	0x31c6
	.long	0x31d6
	.uleb128 0xb
	.long	0xb370
	.uleb128 0xc
	.long	0x300e
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0x26
	.long	.LASF525
	.byte	0xf
	.byte	0xb7
	.long	.LASF526
	.byte	0x3
	.long	0x31ea
	.long	0x31f5
	.uleb128 0xb
	.long	0xb370
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.uleb128 0x20
	.long	.LASF261
	.long	0x2eef
	.byte	0
	.uleb128 0x14
	.long	0x2f5c
	.uleb128 0x6
	.long	.LASF527
	.byte	0x18
	.byte	0xf
	.byte	0xd6
	.long	0x3bf0
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x3191
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x31b3
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x3029
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x305d
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x307a
	.uleb128 0x34
	.long	0x2f5c
	.byte	0
	.byte	0x2
	.uleb128 0xe
	.long	.LASF290
	.byte	0xf
	.byte	0xe2
	.long	0x29
	.byte	0x1
	.uleb128 0xe
	.long	.LASF4
	.byte	0xf
	.byte	0xe3
	.long	0x300e
	.byte	0x1
	.uleb128 0xe
	.long	.LASF10
	.byte	0xf
	.byte	0xe5
	.long	0x8319
	.byte	0x1
	.uleb128 0xe
	.long	.LASF11
	.byte	0xf
	.byte	0xe6
	.long	0x8324
	.byte	0x1
	.uleb128 0xe
	.long	.LASF13
	.byte	0xf
	.byte	0xe7
	.long	0x853e
	.byte	0x1
	.uleb128 0xe
	.long	.LASF14
	.byte	0xf
	.byte	0xe9
	.long	0x8765
	.byte	0x1
	.uleb128 0xe
	.long	.LASF15
	.byte	0xf
	.byte	0xea
	.long	0x3bf0
	.byte	0x1
	.uleb128 0xe
	.long	.LASF16
	.byte	0xf
	.byte	0xeb
	.long	0x3bf5
	.byte	0x1
	.uleb128 0xe
	.long	.LASF5
	.byte	0xf
	.byte	0xec
	.long	0x21c6
	.byte	0x1
	.uleb128 0xe
	.long	.LASF9
	.byte	0xf
	.byte	0xee
	.long	0x2eef
	.byte	0x1
	.uleb128 0x26
	.long	.LASF528
	.byte	0xf
	.byte	0xfd
	.long	.LASF529
	.byte	0x1
	.long	0x32cf
	.long	0x32d5
	.uleb128 0xb
	.long	0xb388
	.byte	0
	.uleb128 0x1d
	.long	.LASF528
	.byte	0xf
	.value	0x108
	.long	.LASF530
	.byte	0x1
	.long	0x32ea
	.long	0x32f5
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0xb38e
	.byte	0
	.uleb128 0x14
	.long	0x32af
	.uleb128 0x1d
	.long	.LASF528
	.byte	0xf
	.value	0x115
	.long	.LASF531
	.byte	0x1
	.long	0x330f
	.long	0x331f
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x32a3
	.uleb128 0xc
	.long	0xb38e
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x121
	.long	.LASF532
	.byte	0x1
	.long	0x3334
	.long	0x3349
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x32a3
	.uleb128 0xc
	.long	0xb394
	.uleb128 0xc
	.long	0xb38e
	.byte	0
	.uleb128 0x14
	.long	0x3243
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x13e
	.long	.LASF533
	.byte	0x1
	.long	0x3363
	.long	0x336e
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0xb39a
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x14f
	.long	.LASF534
	.byte	0x1
	.long	0x3383
	.long	0x338e
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0xb3a0
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x153
	.long	.LASF535
	.byte	0x1
	.long	0x33a3
	.long	0x33b3
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0xb39a
	.uleb128 0xc
	.long	0xb38e
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x15c
	.long	.LASF536
	.byte	0x1
	.long	0x33c8
	.long	0x33d8
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0xb3a0
	.uleb128 0xc
	.long	0xb38e
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x175
	.long	.LASF537
	.byte	0x1
	.long	0x33ed
	.long	0x33fd
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x3bff
	.uleb128 0xc
	.long	0xb38e
	.byte	0
	.uleb128 0x1c
	.long	.LASF538
	.byte	0xf
	.value	0x1a7
	.long	.LASF539
	.byte	0x1
	.long	0x3412
	.long	0x341d
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x27
	.long	.LASF89
	.byte	0x2f
	.byte	0xa7
	.long	.LASF540
	.long	0xb3a6
	.byte	0x1
	.long	0x3435
	.long	0x3440
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0xb39a
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0xf
	.value	0x1c0
	.long	.LASF541
	.long	0xb3a6
	.byte	0x1
	.long	0x3459
	.long	0x3464
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0xb3a0
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0xf
	.value	0x1d6
	.long	.LASF542
	.long	0xb3a6
	.byte	0x1
	.long	0x347d
	.long	0x3488
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x3bff
	.byte	0
	.uleb128 0x1c
	.long	.LASF159
	.byte	0xf
	.value	0x1e8
	.long	.LASF543
	.byte	0x1
	.long	0x349d
	.long	0x34ad
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x32a3
	.uleb128 0xc
	.long	0xb394
	.byte	0
	.uleb128 0x1c
	.long	.LASF159
	.byte	0xf
	.value	0x215
	.long	.LASF544
	.byte	0x1
	.long	0x34c2
	.long	0x34cd
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x3bff
	.byte	0
	.uleb128 0x1e
	.long	.LASF96
	.byte	0xf
	.value	0x223
	.long	.LASF545
	.long	0x3273
	.byte	0x1
	.long	0x34e6
	.long	0x34ec
	.uleb128 0xb
	.long	0xb388
	.byte	0
	.uleb128 0x1e
	.long	.LASF96
	.byte	0xf
	.value	0x22c
	.long	.LASF546
	.long	0x327f
	.byte	0x1
	.long	0x3505
	.long	0x350b
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x1f
	.string	"end"
	.byte	0xf
	.value	0x235
	.long	.LASF547
	.long	0x3273
	.byte	0x1
	.long	0x3524
	.long	0x352a
	.uleb128 0xb
	.long	0xb388
	.byte	0
	.uleb128 0x1f
	.string	"end"
	.byte	0xf
	.value	0x23e
	.long	.LASF548
	.long	0x327f
	.byte	0x1
	.long	0x3543
	.long	0x3549
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x1e
	.long	.LASF101
	.byte	0xf
	.value	0x247
	.long	.LASF549
	.long	0x3297
	.byte	0x1
	.long	0x3562
	.long	0x3568
	.uleb128 0xb
	.long	0xb388
	.byte	0
	.uleb128 0x1e
	.long	.LASF101
	.byte	0xf
	.value	0x250
	.long	.LASF550
	.long	0x328b
	.byte	0x1
	.long	0x3581
	.long	0x3587
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x1e
	.long	.LASF104
	.byte	0xf
	.value	0x259
	.long	.LASF551
	.long	0x3297
	.byte	0x1
	.long	0x35a0
	.long	0x35a6
	.uleb128 0xb
	.long	0xb388
	.byte	0
	.uleb128 0x1e
	.long	.LASF104
	.byte	0xf
	.value	0x262
	.long	.LASF552
	.long	0x328b
	.byte	0x1
	.long	0x35bf
	.long	0x35c5
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x1e
	.long	.LASF107
	.byte	0xf
	.value	0x26c
	.long	.LASF553
	.long	0x327f
	.byte	0x1
	.long	0x35de
	.long	0x35e4
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x1e
	.long	.LASF109
	.byte	0xf
	.value	0x275
	.long	.LASF554
	.long	0x327f
	.byte	0x1
	.long	0x35fd
	.long	0x3603
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x1e
	.long	.LASF111
	.byte	0xf
	.value	0x27e
	.long	.LASF555
	.long	0x328b
	.byte	0x1
	.long	0x361c
	.long	0x3622
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x1e
	.long	.LASF113
	.byte	0xf
	.value	0x287
	.long	.LASF556
	.long	0x328b
	.byte	0x1
	.long	0x363b
	.long	0x3641
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x1e
	.long	.LASF115
	.byte	0xf
	.value	0x28e
	.long	.LASF557
	.long	0x32a3
	.byte	0x1
	.long	0x365a
	.long	0x3660
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x1e
	.long	.LASF119
	.byte	0xf
	.value	0x293
	.long	.LASF558
	.long	0x32a3
	.byte	0x1
	.long	0x3679
	.long	0x367f
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x1c
	.long	.LASF121
	.byte	0xf
	.value	0x2a1
	.long	.LASF559
	.byte	0x1
	.long	0x3694
	.long	0x369f
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x32a3
	.byte	0
	.uleb128 0x1c
	.long	.LASF121
	.byte	0xf
	.value	0x2b5
	.long	.LASF560
	.byte	0x1
	.long	0x36b4
	.long	0x36c4
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x32a3
	.uleb128 0xc
	.long	0xb394
	.byte	0
	.uleb128 0x1c
	.long	.LASF124
	.byte	0xf
	.value	0x2d5
	.long	.LASF561
	.byte	0x1
	.long	0x36d9
	.long	0x36df
	.uleb128 0xb
	.long	0xb388
	.byte	0
	.uleb128 0x1e
	.long	.LASF126
	.byte	0xf
	.value	0x2de
	.long	.LASF562
	.long	0x32a3
	.byte	0x1
	.long	0x36f8
	.long	0x36fe
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x1e
	.long	.LASF132
	.byte	0xf
	.value	0x2e7
	.long	.LASF563
	.long	0x9ef1
	.byte	0x1
	.long	0x3717
	.long	0x371d
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x26
	.long	.LASF128
	.byte	0x2f
	.byte	0x41
	.long	.LASF564
	.byte	0x1
	.long	0x3731
	.long	0x373c
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x32a3
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0xf
	.value	0x30b
	.long	.LASF565
	.long	0x325b
	.byte	0x1
	.long	0x3755
	.long	0x3760
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x32a3
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0xf
	.value	0x31a
	.long	.LASF566
	.long	0x3267
	.byte	0x1
	.long	0x3779
	.long	0x3784
	.uleb128 0xb
	.long	0xb3ac
	.uleb128 0xc
	.long	0x32a3
	.byte	0
	.uleb128 0x1c
	.long	.LASF567
	.byte	0xf
	.value	0x320
	.long	.LASF568
	.byte	0x2
	.long	0x3799
	.long	0x37a4
	.uleb128 0xb
	.long	0xb3ac
	.uleb128 0xc
	.long	0x32a3
	.byte	0
	.uleb128 0x1f
	.string	"at"
	.byte	0xf
	.value	0x336
	.long	.LASF569
	.long	0x325b
	.byte	0x1
	.long	0x37bc
	.long	0x37c7
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x32a3
	.byte	0
	.uleb128 0x1f
	.string	"at"
	.byte	0xf
	.value	0x348
	.long	.LASF570
	.long	0x3267
	.byte	0x1
	.long	0x37df
	.long	0x37ea
	.uleb128 0xb
	.long	0xb3ac
	.uleb128 0xc
	.long	0x32a3
	.byte	0
	.uleb128 0x1e
	.long	.LASF139
	.byte	0xf
	.value	0x353
	.long	.LASF571
	.long	0x325b
	.byte	0x1
	.long	0x3803
	.long	0x3809
	.uleb128 0xb
	.long	0xb388
	.byte	0
	.uleb128 0x1e
	.long	.LASF139
	.byte	0xf
	.value	0x35b
	.long	.LASF572
	.long	0x3267
	.byte	0x1
	.long	0x3822
	.long	0x3828
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x1e
	.long	.LASF142
	.byte	0xf
	.value	0x363
	.long	.LASF573
	.long	0x325b
	.byte	0x1
	.long	0x3841
	.long	0x3847
	.uleb128 0xb
	.long	0xb388
	.byte	0
	.uleb128 0x1e
	.long	.LASF142
	.byte	0xf
	.value	0x36b
	.long	.LASF574
	.long	0x3267
	.byte	0x1
	.long	0x3860
	.long	0x3866
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x1e
	.long	.LASF209
	.byte	0xf
	.value	0x37a
	.long	.LASF575
	.long	0xab81
	.byte	0x1
	.long	0x387f
	.long	0x3885
	.uleb128 0xb
	.long	0xb388
	.byte	0
	.uleb128 0x1e
	.long	.LASF209
	.byte	0xf
	.value	0x382
	.long	.LASF576
	.long	0xb2ff
	.byte	0x1
	.long	0x389e
	.long	0x38a4
	.uleb128 0xb
	.long	0xb3ac
	.byte	0
	.uleb128 0x1c
	.long	.LASF157
	.byte	0xf
	.value	0x391
	.long	.LASF577
	.byte	0x1
	.long	0x38b9
	.long	0x38c4
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0xb394
	.byte	0
	.uleb128 0x1c
	.long	.LASF157
	.byte	0xf
	.value	0x3a3
	.long	.LASF578
	.byte	0x1
	.long	0x38d9
	.long	0x38e4
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0xb3b2
	.byte	0
	.uleb128 0x1c
	.long	.LASF180
	.byte	0xf
	.value	0x3b5
	.long	.LASF579
	.byte	0x1
	.long	0x38f9
	.long	0x38ff
	.uleb128 0xb
	.long	0xb388
	.byte	0
	.uleb128 0x27
	.long	.LASF167
	.byte	0x2f
	.byte	0x6b
	.long	.LASF580
	.long	0x3273
	.byte	0x1
	.long	0x3917
	.long	0x3927
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x327f
	.uleb128 0xc
	.long	0xb394
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0xf
	.value	0x3f6
	.long	.LASF581
	.long	0x3273
	.byte	0x1
	.long	0x3940
	.long	0x3950
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x327f
	.uleb128 0xc
	.long	0xb3b2
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0xf
	.value	0x407
	.long	.LASF582
	.long	0x3273
	.byte	0x1
	.long	0x3969
	.long	0x3979
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x327f
	.uleb128 0xc
	.long	0x3bff
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0xf
	.value	0x41b
	.long	.LASF583
	.long	0x3273
	.byte	0x1
	.long	0x3992
	.long	0x39a7
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x327f
	.uleb128 0xc
	.long	0x32a3
	.uleb128 0xc
	.long	0xb394
	.byte	0
	.uleb128 0x1e
	.long	.LASF176
	.byte	0xf
	.value	0x47a
	.long	.LASF584
	.long	0x3273
	.byte	0x1
	.long	0x39c0
	.long	0x39cb
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x327f
	.byte	0
	.uleb128 0x1e
	.long	.LASF176
	.byte	0xf
	.value	0x495
	.long	.LASF585
	.long	0x3273
	.byte	0x1
	.long	0x39e4
	.long	0x39f4
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x327f
	.uleb128 0xc
	.long	0x327f
	.byte	0
	.uleb128 0x1c
	.long	.LASF205
	.byte	0xf
	.value	0x4aa
	.long	.LASF586
	.byte	0x1
	.long	0x3a09
	.long	0x3a14
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0xb3a6
	.byte	0
	.uleb128 0x1c
	.long	.LASF130
	.byte	0xf
	.value	0x4bb
	.long	.LASF587
	.byte	0x1
	.long	0x3a29
	.long	0x3a2f
	.uleb128 0xb
	.long	0xb388
	.byte	0
	.uleb128 0x1c
	.long	.LASF588
	.byte	0xf
	.value	0x512
	.long	.LASF589
	.byte	0x2
	.long	0x3a44
	.long	0x3a54
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x32a3
	.uleb128 0xc
	.long	0xb394
	.byte	0
	.uleb128 0x1c
	.long	.LASF590
	.byte	0xf
	.value	0x51c
	.long	.LASF591
	.byte	0x2
	.long	0x3a69
	.long	0x3a74
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x32a3
	.byte	0
	.uleb128 0x26
	.long	.LASF592
	.byte	0x2f
	.byte	0xe1
	.long	.LASF593
	.byte	0x2
	.long	0x3a88
	.long	0x3a98
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x21c6
	.uleb128 0xc
	.long	0xb394
	.byte	0
	.uleb128 0x1c
	.long	.LASF594
	.byte	0x2f
	.value	0x1c1
	.long	.LASF595
	.byte	0x2
	.long	0x3aad
	.long	0x3ac2
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x3273
	.uleb128 0xc
	.long	0x32a3
	.uleb128 0xc
	.long	0xb394
	.byte	0
	.uleb128 0x1c
	.long	.LASF596
	.byte	0x2f
	.value	0x21c
	.long	.LASF597
	.byte	0x2
	.long	0x3ad7
	.long	0x3ae2
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x32a3
	.byte	0
	.uleb128 0x1e
	.long	.LASF598
	.byte	0x2f
	.value	0x24e
	.long	.LASF599
	.long	0x9ef1
	.byte	0x2
	.long	0x3afb
	.long	0x3b01
	.uleb128 0xb
	.long	0xb388
	.byte	0
	.uleb128 0x1e
	.long	.LASF600
	.byte	0xf
	.value	0x58e
	.long	.LASF601
	.long	0x32a3
	.byte	0x2
	.long	0x3b1a
	.long	0x3b2a
	.uleb128 0xb
	.long	0xb3ac
	.uleb128 0xc
	.long	0x32a3
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1c
	.long	.LASF602
	.byte	0xf
	.value	0x59c
	.long	.LASF603
	.byte	0x2
	.long	0x3b3f
	.long	0x3b4a
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x324f
	.byte	0
	.uleb128 0x27
	.long	.LASF73
	.byte	0x2f
	.byte	0x8d
	.long	.LASF604
	.long	0x3273
	.byte	0x2
	.long	0x3b62
	.long	0x3b6d
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x3273
	.byte	0
	.uleb128 0x27
	.long	.LASF73
	.byte	0x2f
	.byte	0x99
	.long	.LASF605
	.long	0x3273
	.byte	0x2
	.long	0x3b85
	.long	0x3b95
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0x3273
	.uleb128 0xc
	.long	0x3273
	.byte	0
	.uleb128 0x19
	.long	.LASF606
	.byte	0xf
	.value	0x5ae
	.long	.LASF607
	.long	0x3ba9
	.long	0x3bb9
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0xb3a0
	.uleb128 0xc
	.long	0x22bd
	.byte	0
	.uleb128 0x19
	.long	.LASF606
	.byte	0xf
	.value	0x5b9
	.long	.LASF608
	.long	0x3bcd
	.long	0x3bdd
	.uleb128 0xb
	.long	0xb388
	.uleb128 0xc
	.long	0xb3a0
	.uleb128 0xc
	.long	0x2523
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.uleb128 0x21
	.long	.LASF261
	.long	0x2eef
	.byte	0
	.uleb128 0x2a
	.long	.LASF609
	.uleb128 0x2a
	.long	.LASF610
	.uleb128 0x14
	.long	0x320d
	.uleb128 0x6
	.long	.LASF611
	.byte	0x10
	.byte	0x24
	.byte	0x2f
	.long	0x3ce7
	.uleb128 0xe
	.long	.LASF13
	.byte	0x24
	.byte	0x36
	.long	0xb2ff
	.byte	0x1
	.uleb128 0x9
	.long	.LASF349
	.byte	0x24
	.byte	0x3a
	.long	0x3c0b
	.byte	0
	.uleb128 0xe
	.long	.LASF5
	.byte	0x24
	.byte	0x35
	.long	0x21c6
	.byte	0x1
	.uleb128 0x9
	.long	.LASF350
	.byte	0x24
	.byte	0x3b
	.long	0x3c23
	.byte	0x8
	.uleb128 0xe
	.long	.LASF14
	.byte	0x24
	.byte	0x37
	.long	0xb2ff
	.byte	0x1
	.uleb128 0xa
	.long	.LASF351
	.byte	0x24
	.byte	0x3e
	.long	.LASF612
	.long	0x3c5a
	.long	0x3c6a
	.uleb128 0xb
	.long	0xb3b8
	.uleb128 0xc
	.long	0x3c3b
	.uleb128 0xc
	.long	0x3c23
	.byte	0
	.uleb128 0x26
	.long	.LASF351
	.byte	0x24
	.byte	0x42
	.long	.LASF613
	.byte	0x1
	.long	0x3c7e
	.long	0x3c84
	.uleb128 0xb
	.long	0xb3b8
	.byte	0
	.uleb128 0x27
	.long	.LASF115
	.byte	0x24
	.byte	0x47
	.long	.LASF614
	.long	0x3c23
	.byte	0x1
	.long	0x3c9c
	.long	0x3ca2
	.uleb128 0xb
	.long	0xb3be
	.byte	0
	.uleb128 0x27
	.long	.LASF96
	.byte	0x24
	.byte	0x4b
	.long	.LASF615
	.long	0x3c3b
	.byte	0x1
	.long	0x3cba
	.long	0x3cc0
	.uleb128 0xb
	.long	0xb3be
	.byte	0
	.uleb128 0x38
	.string	"end"
	.byte	0x24
	.byte	0x4f
	.long	.LASF617
	.long	0x3c3b
	.byte	0x1
	.long	0x3cd8
	.long	0x3cde
	.uleb128 0xb
	.long	0xb3be
	.byte	0
	.uleb128 0x2c
	.string	"_E"
	.long	0x29
	.byte	0
	.uleb128 0x14
	.long	0x3bff
	.uleb128 0x7
	.long	.LASF618
	.byte	0x1
	.byte	0x1d
	.byte	0xb2
	.long	0x3d23
	.uleb128 0x16
	.long	.LASF619
	.byte	0x1d
	.byte	0xb6
	.long	0x22b2
	.uleb128 0x16
	.long	.LASF4
	.byte	0x1d
	.byte	0xb7
	.long	0xab81
	.uleb128 0x16
	.long	.LASF10
	.byte	0x1d
	.byte	0xb8
	.long	0xb32e
	.uleb128 0x20
	.long	.LASF620
	.long	0xab81
	.byte	0
	.uleb128 0x47
	.byte	0x30
	.value	0x4d9
	.long	0x2625
	.uleb128 0x23
	.byte	0x31
	.byte	0x33
	.long	0xb40a
	.uleb128 0x36
	.long	.LASF621
	.byte	0x8
	.byte	0xe
	.value	0x418
	.long	0x3f1e
	.uleb128 0x37
	.long	.LASF622
	.byte	0xe
	.value	0x41b
	.long	0xb416
	.uleb128 0x51
	.long	.LASF623
	.byte	0xe
	.value	0x4a7
	.long	0x3d3f
	.byte	0
	.byte	0x3
	.uleb128 0x19
	.long	.LASF624
	.byte	0xe
	.value	0x41d
	.long	.LASF625
	.long	0x3d6d
	.long	0x3d78
	.uleb128 0xb
	.long	0xb41d
	.uleb128 0xc
	.long	0x3d3f
	.byte	0
	.uleb128 0x19
	.long	.LASF624
	.byte	0xe
	.value	0x41f
	.long	.LASF626
	.long	0x3d8c
	.long	0x3d9c
	.uleb128 0xb
	.long	0xb41d
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x52
	.long	.LASF624
	.byte	0xe
	.value	0x429
	.long	.LASF627
	.long	0x3db0
	.long	0x3dbb
	.uleb128 0xb
	.long	0xb41d
	.uleb128 0xc
	.long	0xb423
	.byte	0
	.uleb128 0x52
	.long	.LASF624
	.byte	0xe
	.value	0x42a
	.long	.LASF628
	.long	0x3dcf
	.long	0x3dda
	.uleb128 0xb
	.long	0xb41d
	.uleb128 0xc
	.long	0xb429
	.byte	0
	.uleb128 0x18
	.long	.LASF629
	.byte	0xe
	.value	0x431
	.long	.LASF630
	.long	0x9c79
	.long	0x3df2
	.long	0x3df8
	.uleb128 0xb
	.long	0xb42f
	.byte	0
	.uleb128 0x18
	.long	.LASF631
	.byte	0xe
	.value	0x435
	.long	.LASF632
	.long	0x9c79
	.long	0x3e10
	.long	0x3e16
	.uleb128 0xb
	.long	0xb42f
	.byte	0
	.uleb128 0x19
	.long	.LASF629
	.byte	0xe
	.value	0x447
	.long	.LASF633
	.long	0x3e2a
	.long	0x3e35
	.uleb128 0xb
	.long	0xb41d
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x19
	.long	.LASF631
	.byte	0xe
	.value	0x44a
	.long	.LASF634
	.long	0x3e49
	.long	0x3e54
	.uleb128 0xb
	.long	0xb41d
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x18
	.long	.LASF89
	.byte	0xe
	.value	0x44d
	.long	.LASF635
	.long	0xb435
	.long	0x3e6c
	.long	0x3e77
	.uleb128 0xb
	.long	0xb41d
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x18
	.long	.LASF145
	.byte	0xe
	.value	0x454
	.long	.LASF636
	.long	0xb435
	.long	0x3e8f
	.long	0x3e9a
	.uleb128 0xb
	.long	0xb41d
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x18
	.long	.LASF637
	.byte	0xe
	.value	0x45b
	.long	.LASF638
	.long	0xb435
	.long	0x3eb2
	.long	0x3ebd
	.uleb128 0xb
	.long	0xb41d
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x18
	.long	.LASF639
	.byte	0xe
	.value	0x462
	.long	.LASF640
	.long	0xb435
	.long	0x3ed5
	.long	0x3ee0
	.uleb128 0xb
	.long	0xb41d
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x18
	.long	.LASF641
	.byte	0xe
	.value	0x469
	.long	.LASF642
	.long	0xb435
	.long	0x3ef8
	.long	0x3f03
	.uleb128 0xb
	.long	0xb41d
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x53
	.long	.LASF643
	.byte	0xe
	.value	0x4a4
	.long	.LASF644
	.long	0x3d3f
	.long	0x3f17
	.uleb128 0xb
	.long	0xb42f
	.byte	0
	.byte	0
	.uleb128 0x54
	.long	.LASF645
	.byte	0x10
	.byte	0xe
	.value	0x4ad
	.long	0x4172
	.uleb128 0x42
	.long	.LASF622
	.byte	0xe
	.value	0x4b0
	.long	0xb43b
	.byte	0x1
	.uleb128 0x4b
	.long	.LASF623
	.byte	0xe
	.value	0x53d
	.long	0x3f2b
	.byte	0
	.uleb128 0x1c
	.long	.LASF624
	.byte	0xe
	.value	0x4b2
	.long	.LASF646
	.byte	0x1
	.long	0x3f5a
	.long	0x3f65
	.uleb128 0xb
	.long	0xb442
	.uleb128 0xc
	.long	0x3f2b
	.byte	0
	.uleb128 0x1c
	.long	.LASF624
	.byte	0xe
	.value	0x4b4
	.long	.LASF647
	.byte	0x1
	.long	0x3f7a
	.long	0x3f8a
	.uleb128 0xb
	.long	0xb442
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF624
	.byte	0xe
	.value	0x4be
	.long	.LASF648
	.byte	0x1
	.long	0x3f9f
	.long	0x3faa
	.uleb128 0xb
	.long	0xb442
	.uleb128 0xc
	.long	0xb448
	.byte	0
	.uleb128 0x1d
	.long	.LASF624
	.byte	0xe
	.value	0x4c1
	.long	.LASF649
	.byte	0x1
	.long	0x3fbf
	.long	0x3fca
	.uleb128 0xb
	.long	0xb442
	.uleb128 0xc
	.long	0xb429
	.byte	0
	.uleb128 0x1e
	.long	.LASF629
	.byte	0xe
	.value	0x4c8
	.long	.LASF650
	.long	0x29
	.byte	0x1
	.long	0x3fe3
	.long	0x3fe9
	.uleb128 0xb
	.long	0xb44e
	.byte	0
	.uleb128 0x1e
	.long	.LASF631
	.byte	0xe
	.value	0x4cc
	.long	.LASF651
	.long	0x29
	.byte	0x1
	.long	0x4002
	.long	0x4008
	.uleb128 0xb
	.long	0xb44e
	.byte	0
	.uleb128 0x1c
	.long	.LASF629
	.byte	0xe
	.value	0x4de
	.long	.LASF652
	.byte	0x1
	.long	0x401d
	.long	0x4028
	.uleb128 0xb
	.long	0xb442
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF631
	.byte	0xe
	.value	0x4e1
	.long	.LASF653
	.byte	0x1
	.long	0x403d
	.long	0x4048
	.uleb128 0xb
	.long	0xb442
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0xe
	.value	0x4e4
	.long	.LASF654
	.long	0xb454
	.byte	0x1
	.long	0x4061
	.long	0x406c
	.uleb128 0xb
	.long	0xb442
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1e
	.long	.LASF145
	.byte	0xe
	.value	0x4eb
	.long	.LASF655
	.long	0xb454
	.byte	0x1
	.long	0x4085
	.long	0x4090
	.uleb128 0xb
	.long	0xb442
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1e
	.long	.LASF637
	.byte	0xe
	.value	0x4f2
	.long	.LASF656
	.long	0xb454
	.byte	0x1
	.long	0x40a9
	.long	0x40b4
	.uleb128 0xb
	.long	0xb442
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1e
	.long	.LASF639
	.byte	0xe
	.value	0x4f9
	.long	.LASF657
	.long	0xb454
	.byte	0x1
	.long	0x40cd
	.long	0x40d8
	.uleb128 0xb
	.long	0xb442
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1e
	.long	.LASF641
	.byte	0xe
	.value	0x500
	.long	.LASF658
	.long	0xb454
	.byte	0x1
	.long	0x40f1
	.long	0x40fc
	.uleb128 0xb
	.long	0xb442
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1e
	.long	.LASF643
	.byte	0xe
	.value	0x53a
	.long	.LASF659
	.long	0x3f2b
	.byte	0x1
	.long	0x4115
	.long	0x411b
	.uleb128 0xb
	.long	0xb44e
	.byte	0
	.uleb128 0x1e
	.long	.LASF660
	.byte	0xe
	.value	0x526
	.long	.LASF661
	.long	0xb454
	.byte	0x1
	.long	0x413d
	.long	0x4148
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.uleb128 0xb
	.long	0xb442
	.uleb128 0xc
	.long	0xb423
	.byte	0
	.uleb128 0x55
	.long	.LASF662
	.byte	0xe
	.value	0x531
	.long	.LASF663
	.long	0xb454
	.byte	0x1
	.long	0x4166
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.uleb128 0xb
	.long	0xb442
	.uleb128 0xc
	.long	0xb423
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0x3f1e
	.uleb128 0x54
	.long	.LASF664
	.byte	0x20
	.byte	0xe
	.value	0x543
	.long	0x4371
	.uleb128 0x42
	.long	.LASF622
	.byte	0xe
	.value	0x546
	.long	0xb45a
	.byte	0x1
	.uleb128 0x4b
	.long	.LASF623
	.byte	0xe
	.value	0x5d5
	.long	0x4184
	.byte	0
	.uleb128 0x1c
	.long	.LASF624
	.byte	0xe
	.value	0x548
	.long	.LASF665
	.byte	0x1
	.long	0x41b3
	.long	0x41be
	.uleb128 0xb
	.long	0xb461
	.uleb128 0xc
	.long	0x4184
	.byte	0
	.uleb128 0x1c
	.long	.LASF624
	.byte	0xe
	.value	0x54a
	.long	.LASF666
	.byte	0x1
	.long	0x41d3
	.long	0x41e3
	.uleb128 0xb
	.long	0xb461
	.uleb128 0xc
	.long	0x9e79
	.uleb128 0xc
	.long	0x9e79
	.byte	0
	.uleb128 0x1c
	.long	.LASF624
	.byte	0xe
	.value	0x555
	.long	.LASF667
	.byte	0x1
	.long	0x41f8
	.long	0x4203
	.uleb128 0xb
	.long	0xb461
	.uleb128 0xc
	.long	0xb448
	.byte	0
	.uleb128 0x1c
	.long	.LASF624
	.byte	0xe
	.value	0x558
	.long	.LASF668
	.byte	0x1
	.long	0x4218
	.long	0x4223
	.uleb128 0xb
	.long	0xb461
	.uleb128 0xc
	.long	0xb423
	.byte	0
	.uleb128 0x1e
	.long	.LASF629
	.byte	0xe
	.value	0x560
	.long	.LASF669
	.long	0x9e79
	.byte	0x1
	.long	0x423c
	.long	0x4242
	.uleb128 0xb
	.long	0xb467
	.byte	0
	.uleb128 0x1e
	.long	.LASF631
	.byte	0xe
	.value	0x564
	.long	.LASF670
	.long	0x9e79
	.byte	0x1
	.long	0x425b
	.long	0x4261
	.uleb128 0xb
	.long	0xb467
	.byte	0
	.uleb128 0x1c
	.long	.LASF629
	.byte	0xe
	.value	0x576
	.long	.LASF671
	.byte	0x1
	.long	0x4276
	.long	0x4281
	.uleb128 0xb
	.long	0xb461
	.uleb128 0xc
	.long	0x9e79
	.byte	0
	.uleb128 0x1c
	.long	.LASF631
	.byte	0xe
	.value	0x579
	.long	.LASF672
	.byte	0x1
	.long	0x4296
	.long	0x42a1
	.uleb128 0xb
	.long	0xb461
	.uleb128 0xc
	.long	0x9e79
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0xe
	.value	0x57c
	.long	.LASF673
	.long	0xb46d
	.byte	0x1
	.long	0x42ba
	.long	0x42c5
	.uleb128 0xb
	.long	0xb461
	.uleb128 0xc
	.long	0x9e79
	.byte	0
	.uleb128 0x1e
	.long	.LASF145
	.byte	0xe
	.value	0x583
	.long	.LASF674
	.long	0xb46d
	.byte	0x1
	.long	0x42de
	.long	0x42e9
	.uleb128 0xb
	.long	0xb461
	.uleb128 0xc
	.long	0x9e79
	.byte	0
	.uleb128 0x1e
	.long	.LASF637
	.byte	0xe
	.value	0x58a
	.long	.LASF675
	.long	0xb46d
	.byte	0x1
	.long	0x4302
	.long	0x430d
	.uleb128 0xb
	.long	0xb461
	.uleb128 0xc
	.long	0x9e79
	.byte	0
	.uleb128 0x1e
	.long	.LASF639
	.byte	0xe
	.value	0x591
	.long	.LASF676
	.long	0xb46d
	.byte	0x1
	.long	0x4326
	.long	0x4331
	.uleb128 0xb
	.long	0xb461
	.uleb128 0xc
	.long	0x9e79
	.byte	0
	.uleb128 0x1e
	.long	.LASF641
	.byte	0xe
	.value	0x598
	.long	.LASF677
	.long	0xb46d
	.byte	0x1
	.long	0x434a
	.long	0x4355
	.uleb128 0xb
	.long	0xb461
	.uleb128 0xc
	.long	0x9e79
	.byte	0
	.uleb128 0x55
	.long	.LASF643
	.byte	0xe
	.value	0x5d2
	.long	.LASF678
	.long	0x4184
	.byte	0x1
	.long	0x436a
	.uleb128 0xb
	.long	0xb467
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0x4177
	.uleb128 0x14
	.long	0x3d32
	.uleb128 0x2b
	.long	.LASF679
	.byte	0x1
	.byte	0x32
	.byte	0x27
	.uleb128 0x56
	.long	.LASF680
	.byte	0x1
	.byte	0x33
	.value	0x472
	.uleb128 0x36
	.long	.LASF681
	.byte	0x1
	.byte	0x23
	.value	0x1ba
	.long	0x4487
	.uleb128 0x37
	.long	.LASF9
	.byte	0x23
	.value	0x1bd
	.long	0x4487
	.uleb128 0x37
	.long	.LASF290
	.byte	0x23
	.value	0x1bf
	.long	0x912d
	.uleb128 0x37
	.long	.LASF4
	.byte	0x23
	.value	0x1c2
	.long	0x14128
	.uleb128 0x37
	.long	.LASF334
	.byte	0x23
	.value	0x1cb
	.long	0xa1f5
	.uleb128 0x37
	.long	.LASF5
	.byte	0x23
	.value	0x1d1
	.long	0x21c6
	.uleb128 0x1b
	.long	.LASF335
	.byte	0x23
	.value	0x1ea
	.long	.LASF682
	.long	0x43b1
	.long	0x43f4
	.uleb128 0xc
	.long	0x14139
	.uleb128 0xc
	.long	0x43c9
	.byte	0
	.uleb128 0x1b
	.long	.LASF335
	.byte	0x23
	.value	0x1f8
	.long	.LASF683
	.long	0x43b1
	.long	0x4418
	.uleb128 0xc
	.long	0x14139
	.uleb128 0xc
	.long	0x43c9
	.uleb128 0xc
	.long	0x43bd
	.byte	0
	.uleb128 0x1a
	.long	.LASF338
	.byte	0x23
	.value	0x204
	.long	.LASF684
	.long	0x4438
	.uleb128 0xc
	.long	0x14139
	.uleb128 0xc
	.long	0x43b1
	.uleb128 0xc
	.long	0x43c9
	.byte	0
	.uleb128 0x1b
	.long	.LASF119
	.byte	0x23
	.value	0x226
	.long	.LASF685
	.long	0x43c9
	.long	0x4452
	.uleb128 0xc
	.long	0x1413f
	.byte	0
	.uleb128 0x14
	.long	0x4399
	.uleb128 0x1b
	.long	.LASF341
	.byte	0x23
	.value	0x22f
	.long	.LASF686
	.long	0x4399
	.long	0x4471
	.uleb128 0xc
	.long	0x1413f
	.byte	0
	.uleb128 0x37
	.long	.LASF343
	.byte	0x23
	.value	0x1dd
	.long	0x4487
	.uleb128 0x20
	.long	.LASF261
	.long	0x4487
	.byte	0
	.uleb128 0x6
	.long	.LASF687
	.byte	0x1
	.byte	0x21
	.byte	0x5c
	.long	0x44ef
	.uleb128 0x34
	.long	0x887f
	.byte	0
	.byte	0x1
	.uleb128 0x26
	.long	.LASF328
	.byte	0x21
	.byte	0x71
	.long	.LASF688
	.byte	0x1
	.long	0x44ae
	.long	0x44b4
	.uleb128 0xb
	.long	0x1417b
	.byte	0
	.uleb128 0x26
	.long	.LASF328
	.byte	0x21
	.byte	0x73
	.long	.LASF689
	.byte	0x1
	.long	0x44c8
	.long	0x44d3
	.uleb128 0xb
	.long	0x1417b
	.uleb128 0xc
	.long	0x14151
	.byte	0
	.uleb128 0x35
	.long	.LASF331
	.byte	0x21
	.byte	0x79
	.long	.LASF690
	.byte	0x1
	.long	0x44e3
	.uleb128 0xb
	.long	0x1417b
	.uleb128 0xb
	.long	0x30
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0x4487
	.uleb128 0x7
	.long	.LASF691
	.byte	0x18
	.byte	0xf
	.byte	0x48
	.long	0x47bc
	.uleb128 0x7
	.long	.LASF496
	.byte	0x18
	.byte	0xf
	.byte	0x4f
	.long	0x45c2
	.uleb128 0x8
	.long	0x4487
	.byte	0
	.uleb128 0x9
	.long	.LASF497
	.byte	0xf
	.byte	0x52
	.long	0x45c2
	.byte	0
	.uleb128 0x9
	.long	.LASF498
	.byte	0xf
	.byte	0x53
	.long	0x45c2
	.byte	0x8
	.uleb128 0x9
	.long	.LASF499
	.byte	0xf
	.byte	0x54
	.long	0x45c2
	.byte	0x10
	.uleb128 0xa
	.long	.LASF496
	.byte	0xf
	.byte	0x56
	.long	.LASF692
	.long	0x4549
	.long	0x454f
	.uleb128 0xb
	.long	0x14181
	.byte	0
	.uleb128 0xa
	.long	.LASF496
	.byte	0xf
	.byte	0x5a
	.long	.LASF693
	.long	0x4562
	.long	0x456d
	.uleb128 0xb
	.long	0x14181
	.uleb128 0xc
	.long	0x14187
	.byte	0
	.uleb128 0xa
	.long	.LASF496
	.byte	0xf
	.byte	0x5f
	.long	.LASF694
	.long	0x4580
	.long	0x458b
	.uleb128 0xb
	.long	0x14181
	.uleb128 0xc
	.long	0x1418d
	.byte	0
	.uleb128 0xa
	.long	.LASF503
	.byte	0xf
	.byte	0x65
	.long	.LASF695
	.long	0x459e
	.long	0x45a9
	.uleb128 0xb
	.long	0x14181
	.uleb128 0xc
	.long	0x14193
	.byte	0
	.uleb128 0xd
	.long	.LASF696
	.long	.LASF698
	.long	0x45b6
	.uleb128 0xb
	.long	0x14181
	.uleb128 0xb
	.long	0x30
	.byte	0
	.byte	0
	.uleb128 0x16
	.long	.LASF4
	.byte	0xf
	.byte	0x4d
	.long	0x87a1
	.uleb128 0x16
	.long	.LASF505
	.byte	0xf
	.byte	0x4b
	.long	0x8860
	.uleb128 0x14
	.long	0x45cd
	.uleb128 0x9
	.long	.LASF506
	.byte	0xf
	.byte	0xa4
	.long	0x4500
	.byte	0
	.uleb128 0x16
	.long	.LASF9
	.byte	0xf
	.byte	0x6e
	.long	0x4487
	.uleb128 0x17
	.long	.LASF507
	.byte	0xf
	.byte	0x71
	.long	.LASF699
	.long	0x14199
	.long	0x460b
	.long	0x4611
	.uleb128 0xb
	.long	0x1419f
	.byte	0
	.uleb128 0x17
	.long	.LASF507
	.byte	0xf
	.byte	0x75
	.long	.LASF700
	.long	0x14187
	.long	0x4628
	.long	0x462e
	.uleb128 0xb
	.long	0x141a5
	.byte	0
	.uleb128 0x17
	.long	.LASF211
	.byte	0xf
	.byte	0x79
	.long	.LASF701
	.long	0x45e9
	.long	0x4645
	.long	0x464b
	.uleb128 0xb
	.long	0x141a5
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x7c
	.long	.LASF702
	.long	0x465e
	.long	0x4664
	.uleb128 0xb
	.long	0x1419f
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x7f
	.long	.LASF703
	.long	0x4677
	.long	0x4682
	.uleb128 0xb
	.long	0x1419f
	.uleb128 0xc
	.long	0x141ab
	.byte	0
	.uleb128 0x14
	.long	0x45e9
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x82
	.long	.LASF704
	.long	0x469a
	.long	0x46a5
	.uleb128 0xb
	.long	0x1419f
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x86
	.long	.LASF705
	.long	0x46b8
	.long	0x46c8
	.uleb128 0xb
	.long	0x1419f
	.uleb128 0xc
	.long	0x21c6
	.uleb128 0xc
	.long	0x141ab
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x8b
	.long	.LASF706
	.long	0x46db
	.long	0x46e6
	.uleb128 0xb
	.long	0x1419f
	.uleb128 0xc
	.long	0x1418d
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x8e
	.long	.LASF707
	.long	0x46f9
	.long	0x4704
	.uleb128 0xb
	.long	0x1419f
	.uleb128 0xc
	.long	0x141b1
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x92
	.long	.LASF708
	.long	0x4717
	.long	0x4727
	.uleb128 0xb
	.long	0x1419f
	.uleb128 0xc
	.long	0x141b1
	.uleb128 0xc
	.long	0x141ab
	.byte	0
	.uleb128 0xa
	.long	.LASF519
	.byte	0xf
	.byte	0x9f
	.long	.LASF709
	.long	0x473a
	.long	0x4745
	.uleb128 0xb
	.long	0x1419f
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x17
	.long	.LASF521
	.byte	0xf
	.byte	0xa7
	.long	.LASF710
	.long	0x45c2
	.long	0x475c
	.long	0x4767
	.uleb128 0xb
	.long	0x1419f
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0xa
	.long	.LASF523
	.byte	0xf
	.byte	0xae
	.long	.LASF711
	.long	0x477a
	.long	0x478a
	.uleb128 0xb
	.long	0x1419f
	.uleb128 0xc
	.long	0x45c2
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0x26
	.long	.LASF525
	.byte	0xf
	.byte	0xb7
	.long	.LASF712
	.byte	0x3
	.long	0x479e
	.long	0x47a9
	.uleb128 0xb
	.long	0x1419f
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x912d
	.uleb128 0x20
	.long	.LASF261
	.long	0x4487
	.byte	0
	.uleb128 0x14
	.long	0x44f4
	.uleb128 0x6
	.long	.LASF713
	.byte	0x18
	.byte	0xf
	.byte	0xd6
	.long	0x51a4
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x4745
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x4767
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x45dd
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x4611
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x462e
	.uleb128 0x34
	.long	0x44f4
	.byte	0
	.byte	0x2
	.uleb128 0xe
	.long	.LASF290
	.byte	0xf
	.byte	0xe2
	.long	0x912d
	.byte	0x1
	.uleb128 0xe
	.long	.LASF4
	.byte	0xf
	.byte	0xe3
	.long	0x45c2
	.byte	0x1
	.uleb128 0xe
	.long	.LASF10
	.byte	0xf
	.byte	0xe5
	.long	0x87ac
	.byte	0x1
	.uleb128 0xe
	.long	.LASF11
	.byte	0xf
	.byte	0xe6
	.long	0x87b7
	.byte	0x1
	.uleb128 0xe
	.long	.LASF13
	.byte	0xf
	.byte	0xe7
	.long	0x89d1
	.byte	0x1
	.uleb128 0xe
	.long	.LASF14
	.byte	0xf
	.byte	0xe9
	.long	0x89d6
	.byte	0x1
	.uleb128 0xe
	.long	.LASF15
	.byte	0xf
	.byte	0xea
	.long	0x51a4
	.byte	0x1
	.uleb128 0xe
	.long	.LASF16
	.byte	0xf
	.byte	0xeb
	.long	0x51a9
	.byte	0x1
	.uleb128 0xe
	.long	.LASF5
	.byte	0xf
	.byte	0xec
	.long	0x21c6
	.byte	0x1
	.uleb128 0xe
	.long	.LASF9
	.byte	0xf
	.byte	0xee
	.long	0x4487
	.byte	0x1
	.uleb128 0x26
	.long	.LASF528
	.byte	0xf
	.byte	0xfd
	.long	.LASF714
	.byte	0x1
	.long	0x4883
	.long	0x4889
	.uleb128 0xb
	.long	0x141b7
	.byte	0
	.uleb128 0x1d
	.long	.LASF528
	.byte	0xf
	.value	0x108
	.long	.LASF715
	.byte	0x1
	.long	0x489e
	.long	0x48a9
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x141bd
	.byte	0
	.uleb128 0x14
	.long	0x4863
	.uleb128 0x1d
	.long	.LASF528
	.byte	0xf
	.value	0x115
	.long	.LASF716
	.byte	0x1
	.long	0x48c3
	.long	0x48d3
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4857
	.uleb128 0xc
	.long	0x141bd
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x121
	.long	.LASF717
	.byte	0x1
	.long	0x48e8
	.long	0x48fd
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4857
	.uleb128 0xc
	.long	0x141c3
	.uleb128 0xc
	.long	0x141bd
	.byte	0
	.uleb128 0x14
	.long	0x47f7
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x13e
	.long	.LASF718
	.byte	0x1
	.long	0x4917
	.long	0x4922
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x141c9
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x14f
	.long	.LASF719
	.byte	0x1
	.long	0x4937
	.long	0x4942
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x141cf
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x153
	.long	.LASF720
	.byte	0x1
	.long	0x4957
	.long	0x4967
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x141c9
	.uleb128 0xc
	.long	0x141bd
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x15c
	.long	.LASF721
	.byte	0x1
	.long	0x497c
	.long	0x498c
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x141cf
	.uleb128 0xc
	.long	0x141bd
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x175
	.long	.LASF722
	.byte	0x1
	.long	0x49a1
	.long	0x49b1
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x51b3
	.uleb128 0xc
	.long	0x141bd
	.byte	0
	.uleb128 0x1c
	.long	.LASF538
	.byte	0xf
	.value	0x1a7
	.long	.LASF723
	.byte	0x1
	.long	0x49c6
	.long	0x49d1
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x27
	.long	.LASF89
	.byte	0x2f
	.byte	0xa7
	.long	.LASF724
	.long	0x141d5
	.byte	0x1
	.long	0x49e9
	.long	0x49f4
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x141c9
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0xf
	.value	0x1c0
	.long	.LASF725
	.long	0x141d5
	.byte	0x1
	.long	0x4a0d
	.long	0x4a18
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x141cf
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0xf
	.value	0x1d6
	.long	.LASF726
	.long	0x141d5
	.byte	0x1
	.long	0x4a31
	.long	0x4a3c
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x51b3
	.byte	0
	.uleb128 0x1c
	.long	.LASF159
	.byte	0xf
	.value	0x1e8
	.long	.LASF727
	.byte	0x1
	.long	0x4a51
	.long	0x4a61
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4857
	.uleb128 0xc
	.long	0x141c3
	.byte	0
	.uleb128 0x1c
	.long	.LASF159
	.byte	0xf
	.value	0x215
	.long	.LASF728
	.byte	0x1
	.long	0x4a76
	.long	0x4a81
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x51b3
	.byte	0
	.uleb128 0x1e
	.long	.LASF96
	.byte	0xf
	.value	0x223
	.long	.LASF729
	.long	0x4827
	.byte	0x1
	.long	0x4a9a
	.long	0x4aa0
	.uleb128 0xb
	.long	0x141b7
	.byte	0
	.uleb128 0x1e
	.long	.LASF96
	.byte	0xf
	.value	0x22c
	.long	.LASF730
	.long	0x4833
	.byte	0x1
	.long	0x4ab9
	.long	0x4abf
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x1f
	.string	"end"
	.byte	0xf
	.value	0x235
	.long	.LASF731
	.long	0x4827
	.byte	0x1
	.long	0x4ad8
	.long	0x4ade
	.uleb128 0xb
	.long	0x141b7
	.byte	0
	.uleb128 0x1f
	.string	"end"
	.byte	0xf
	.value	0x23e
	.long	.LASF732
	.long	0x4833
	.byte	0x1
	.long	0x4af7
	.long	0x4afd
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x1e
	.long	.LASF101
	.byte	0xf
	.value	0x247
	.long	.LASF733
	.long	0x484b
	.byte	0x1
	.long	0x4b16
	.long	0x4b1c
	.uleb128 0xb
	.long	0x141b7
	.byte	0
	.uleb128 0x1e
	.long	.LASF101
	.byte	0xf
	.value	0x250
	.long	.LASF734
	.long	0x483f
	.byte	0x1
	.long	0x4b35
	.long	0x4b3b
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x1e
	.long	.LASF104
	.byte	0xf
	.value	0x259
	.long	.LASF735
	.long	0x484b
	.byte	0x1
	.long	0x4b54
	.long	0x4b5a
	.uleb128 0xb
	.long	0x141b7
	.byte	0
	.uleb128 0x1e
	.long	.LASF104
	.byte	0xf
	.value	0x262
	.long	.LASF736
	.long	0x483f
	.byte	0x1
	.long	0x4b73
	.long	0x4b79
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x1e
	.long	.LASF107
	.byte	0xf
	.value	0x26c
	.long	.LASF737
	.long	0x4833
	.byte	0x1
	.long	0x4b92
	.long	0x4b98
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x1e
	.long	.LASF109
	.byte	0xf
	.value	0x275
	.long	.LASF738
	.long	0x4833
	.byte	0x1
	.long	0x4bb1
	.long	0x4bb7
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x1e
	.long	.LASF111
	.byte	0xf
	.value	0x27e
	.long	.LASF739
	.long	0x483f
	.byte	0x1
	.long	0x4bd0
	.long	0x4bd6
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x1e
	.long	.LASF113
	.byte	0xf
	.value	0x287
	.long	.LASF740
	.long	0x483f
	.byte	0x1
	.long	0x4bef
	.long	0x4bf5
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x1e
	.long	.LASF115
	.byte	0xf
	.value	0x28e
	.long	.LASF741
	.long	0x4857
	.byte	0x1
	.long	0x4c0e
	.long	0x4c14
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x1e
	.long	.LASF119
	.byte	0xf
	.value	0x293
	.long	.LASF742
	.long	0x4857
	.byte	0x1
	.long	0x4c2d
	.long	0x4c33
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x1c
	.long	.LASF121
	.byte	0xf
	.value	0x2a1
	.long	.LASF743
	.byte	0x1
	.long	0x4c48
	.long	0x4c53
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4857
	.byte	0
	.uleb128 0x1c
	.long	.LASF121
	.byte	0xf
	.value	0x2b5
	.long	.LASF744
	.byte	0x1
	.long	0x4c68
	.long	0x4c78
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4857
	.uleb128 0xc
	.long	0x141c3
	.byte	0
	.uleb128 0x1c
	.long	.LASF124
	.byte	0xf
	.value	0x2d5
	.long	.LASF745
	.byte	0x1
	.long	0x4c8d
	.long	0x4c93
	.uleb128 0xb
	.long	0x141b7
	.byte	0
	.uleb128 0x1e
	.long	.LASF126
	.byte	0xf
	.value	0x2de
	.long	.LASF746
	.long	0x4857
	.byte	0x1
	.long	0x4cac
	.long	0x4cb2
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x1e
	.long	.LASF132
	.byte	0xf
	.value	0x2e7
	.long	.LASF747
	.long	0x9ef1
	.byte	0x1
	.long	0x4ccb
	.long	0x4cd1
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x26
	.long	.LASF128
	.byte	0x2f
	.byte	0x41
	.long	.LASF748
	.byte	0x1
	.long	0x4ce5
	.long	0x4cf0
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4857
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0xf
	.value	0x30b
	.long	.LASF749
	.long	0x480f
	.byte	0x1
	.long	0x4d09
	.long	0x4d14
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4857
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0xf
	.value	0x31a
	.long	.LASF750
	.long	0x481b
	.byte	0x1
	.long	0x4d2d
	.long	0x4d38
	.uleb128 0xb
	.long	0x141db
	.uleb128 0xc
	.long	0x4857
	.byte	0
	.uleb128 0x1c
	.long	.LASF567
	.byte	0xf
	.value	0x320
	.long	.LASF751
	.byte	0x2
	.long	0x4d4d
	.long	0x4d58
	.uleb128 0xb
	.long	0x141db
	.uleb128 0xc
	.long	0x4857
	.byte	0
	.uleb128 0x1f
	.string	"at"
	.byte	0xf
	.value	0x336
	.long	.LASF752
	.long	0x480f
	.byte	0x1
	.long	0x4d70
	.long	0x4d7b
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4857
	.byte	0
	.uleb128 0x1f
	.string	"at"
	.byte	0xf
	.value	0x348
	.long	.LASF753
	.long	0x481b
	.byte	0x1
	.long	0x4d93
	.long	0x4d9e
	.uleb128 0xb
	.long	0x141db
	.uleb128 0xc
	.long	0x4857
	.byte	0
	.uleb128 0x1e
	.long	.LASF139
	.byte	0xf
	.value	0x353
	.long	.LASF754
	.long	0x480f
	.byte	0x1
	.long	0x4db7
	.long	0x4dbd
	.uleb128 0xb
	.long	0x141b7
	.byte	0
	.uleb128 0x1e
	.long	.LASF139
	.byte	0xf
	.value	0x35b
	.long	.LASF755
	.long	0x481b
	.byte	0x1
	.long	0x4dd6
	.long	0x4ddc
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x1e
	.long	.LASF142
	.byte	0xf
	.value	0x363
	.long	.LASF756
	.long	0x480f
	.byte	0x1
	.long	0x4df5
	.long	0x4dfb
	.uleb128 0xb
	.long	0x141b7
	.byte	0
	.uleb128 0x1e
	.long	.LASF142
	.byte	0xf
	.value	0x36b
	.long	.LASF757
	.long	0x481b
	.byte	0x1
	.long	0x4e14
	.long	0x4e1a
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x1e
	.long	.LASF209
	.byte	0xf
	.value	0x37a
	.long	.LASF758
	.long	0x14128
	.byte	0x1
	.long	0x4e33
	.long	0x4e39
	.uleb128 0xb
	.long	0x141b7
	.byte	0
	.uleb128 0x1e
	.long	.LASF209
	.byte	0xf
	.value	0x382
	.long	.LASF759
	.long	0x1412e
	.byte	0x1
	.long	0x4e52
	.long	0x4e58
	.uleb128 0xb
	.long	0x141db
	.byte	0
	.uleb128 0x1c
	.long	.LASF157
	.byte	0xf
	.value	0x391
	.long	.LASF760
	.byte	0x1
	.long	0x4e6d
	.long	0x4e78
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x141c3
	.byte	0
	.uleb128 0x1c
	.long	.LASF157
	.byte	0xf
	.value	0x3a3
	.long	.LASF761
	.byte	0x1
	.long	0x4e8d
	.long	0x4e98
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x141e1
	.byte	0
	.uleb128 0x1c
	.long	.LASF180
	.byte	0xf
	.value	0x3b5
	.long	.LASF762
	.byte	0x1
	.long	0x4ead
	.long	0x4eb3
	.uleb128 0xb
	.long	0x141b7
	.byte	0
	.uleb128 0x27
	.long	.LASF167
	.byte	0x2f
	.byte	0x6b
	.long	.LASF763
	.long	0x4827
	.byte	0x1
	.long	0x4ecb
	.long	0x4edb
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4833
	.uleb128 0xc
	.long	0x141c3
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0xf
	.value	0x3f6
	.long	.LASF764
	.long	0x4827
	.byte	0x1
	.long	0x4ef4
	.long	0x4f04
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4833
	.uleb128 0xc
	.long	0x141e1
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0xf
	.value	0x407
	.long	.LASF765
	.long	0x4827
	.byte	0x1
	.long	0x4f1d
	.long	0x4f2d
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4833
	.uleb128 0xc
	.long	0x51b3
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0xf
	.value	0x41b
	.long	.LASF766
	.long	0x4827
	.byte	0x1
	.long	0x4f46
	.long	0x4f5b
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4833
	.uleb128 0xc
	.long	0x4857
	.uleb128 0xc
	.long	0x141c3
	.byte	0
	.uleb128 0x1e
	.long	.LASF176
	.byte	0xf
	.value	0x47a
	.long	.LASF767
	.long	0x4827
	.byte	0x1
	.long	0x4f74
	.long	0x4f7f
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4833
	.byte	0
	.uleb128 0x1e
	.long	.LASF176
	.byte	0xf
	.value	0x495
	.long	.LASF768
	.long	0x4827
	.byte	0x1
	.long	0x4f98
	.long	0x4fa8
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4833
	.uleb128 0xc
	.long	0x4833
	.byte	0
	.uleb128 0x1c
	.long	.LASF205
	.byte	0xf
	.value	0x4aa
	.long	.LASF769
	.byte	0x1
	.long	0x4fbd
	.long	0x4fc8
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x141d5
	.byte	0
	.uleb128 0x1c
	.long	.LASF130
	.byte	0xf
	.value	0x4bb
	.long	.LASF770
	.byte	0x1
	.long	0x4fdd
	.long	0x4fe3
	.uleb128 0xb
	.long	0x141b7
	.byte	0
	.uleb128 0x1c
	.long	.LASF588
	.byte	0xf
	.value	0x512
	.long	.LASF771
	.byte	0x2
	.long	0x4ff8
	.long	0x5008
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4857
	.uleb128 0xc
	.long	0x141c3
	.byte	0
	.uleb128 0x1c
	.long	.LASF590
	.byte	0xf
	.value	0x51c
	.long	.LASF772
	.byte	0x2
	.long	0x501d
	.long	0x5028
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4857
	.byte	0
	.uleb128 0x26
	.long	.LASF592
	.byte	0x2f
	.byte	0xe1
	.long	.LASF773
	.byte	0x2
	.long	0x503c
	.long	0x504c
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x21c6
	.uleb128 0xc
	.long	0x141c3
	.byte	0
	.uleb128 0x1c
	.long	.LASF594
	.byte	0x2f
	.value	0x1c1
	.long	.LASF774
	.byte	0x2
	.long	0x5061
	.long	0x5076
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4827
	.uleb128 0xc
	.long	0x4857
	.uleb128 0xc
	.long	0x141c3
	.byte	0
	.uleb128 0x1c
	.long	.LASF596
	.byte	0x2f
	.value	0x21c
	.long	.LASF775
	.byte	0x2
	.long	0x508b
	.long	0x5096
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4857
	.byte	0
	.uleb128 0x1e
	.long	.LASF598
	.byte	0x2f
	.value	0x24e
	.long	.LASF776
	.long	0x9ef1
	.byte	0x2
	.long	0x50af
	.long	0x50b5
	.uleb128 0xb
	.long	0x141b7
	.byte	0
	.uleb128 0x1e
	.long	.LASF600
	.byte	0xf
	.value	0x58e
	.long	.LASF777
	.long	0x4857
	.byte	0x2
	.long	0x50ce
	.long	0x50de
	.uleb128 0xb
	.long	0x141db
	.uleb128 0xc
	.long	0x4857
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1c
	.long	.LASF602
	.byte	0xf
	.value	0x59c
	.long	.LASF778
	.byte	0x2
	.long	0x50f3
	.long	0x50fe
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4803
	.byte	0
	.uleb128 0x27
	.long	.LASF73
	.byte	0x2f
	.byte	0x8d
	.long	.LASF779
	.long	0x4827
	.byte	0x2
	.long	0x5116
	.long	0x5121
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4827
	.byte	0
	.uleb128 0x27
	.long	.LASF73
	.byte	0x2f
	.byte	0x99
	.long	.LASF780
	.long	0x4827
	.byte	0x2
	.long	0x5139
	.long	0x5149
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x4827
	.uleb128 0xc
	.long	0x4827
	.byte	0
	.uleb128 0x19
	.long	.LASF606
	.byte	0xf
	.value	0x5ae
	.long	.LASF781
	.long	0x515d
	.long	0x516d
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x141cf
	.uleb128 0xc
	.long	0x22bd
	.byte	0
	.uleb128 0x19
	.long	.LASF606
	.byte	0xf
	.value	0x5b9
	.long	.LASF782
	.long	0x5181
	.long	0x5191
	.uleb128 0xb
	.long	0x141b7
	.uleb128 0xc
	.long	0x141cf
	.uleb128 0xc
	.long	0x2523
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x912d
	.uleb128 0x21
	.long	.LASF261
	.long	0x4487
	.byte	0
	.uleb128 0x2a
	.long	.LASF783
	.uleb128 0x2a
	.long	.LASF784
	.uleb128 0x14
	.long	0x47c1
	.uleb128 0x6
	.long	.LASF785
	.byte	0x10
	.byte	0x24
	.byte	0x2f
	.long	0x529b
	.uleb128 0xe
	.long	.LASF13
	.byte	0x24
	.byte	0x36
	.long	0x1412e
	.byte	0x1
	.uleb128 0x9
	.long	.LASF349
	.byte	0x24
	.byte	0x3a
	.long	0x51bf
	.byte	0
	.uleb128 0xe
	.long	.LASF5
	.byte	0x24
	.byte	0x35
	.long	0x21c6
	.byte	0x1
	.uleb128 0x9
	.long	.LASF350
	.byte	0x24
	.byte	0x3b
	.long	0x51d7
	.byte	0x8
	.uleb128 0xe
	.long	.LASF14
	.byte	0x24
	.byte	0x37
	.long	0x1412e
	.byte	0x1
	.uleb128 0xa
	.long	.LASF351
	.byte	0x24
	.byte	0x3e
	.long	.LASF786
	.long	0x520e
	.long	0x521e
	.uleb128 0xb
	.long	0x1484e
	.uleb128 0xc
	.long	0x51ef
	.uleb128 0xc
	.long	0x51d7
	.byte	0
	.uleb128 0x26
	.long	.LASF351
	.byte	0x24
	.byte	0x42
	.long	.LASF787
	.byte	0x1
	.long	0x5232
	.long	0x5238
	.uleb128 0xb
	.long	0x1484e
	.byte	0
	.uleb128 0x27
	.long	.LASF115
	.byte	0x24
	.byte	0x47
	.long	.LASF788
	.long	0x51d7
	.byte	0x1
	.long	0x5250
	.long	0x5256
	.uleb128 0xb
	.long	0x14854
	.byte	0
	.uleb128 0x27
	.long	.LASF96
	.byte	0x24
	.byte	0x4b
	.long	.LASF789
	.long	0x51ef
	.byte	0x1
	.long	0x526e
	.long	0x5274
	.uleb128 0xb
	.long	0x14854
	.byte	0
	.uleb128 0x38
	.string	"end"
	.byte	0x24
	.byte	0x4f
	.long	.LASF790
	.long	0x51ef
	.byte	0x1
	.long	0x528c
	.long	0x5292
	.uleb128 0xb
	.long	0x14854
	.byte	0
	.uleb128 0x2c
	.string	"_E"
	.long	0x912d
	.byte	0
	.uleb128 0x36
	.long	.LASF791
	.byte	0x1
	.byte	0x23
	.value	0x1ba
	.long	0x5396
	.uleb128 0x37
	.long	.LASF9
	.byte	0x23
	.value	0x1bd
	.long	0x5396
	.uleb128 0x37
	.long	.LASF290
	.byte	0x23
	.value	0x1bf
	.long	0x30
	.uleb128 0x37
	.long	.LASF4
	.byte	0x23
	.value	0x1c2
	.long	0xaa8a
	.uleb128 0x37
	.long	.LASF334
	.byte	0x23
	.value	0x1cb
	.long	0xa1f5
	.uleb128 0x37
	.long	.LASF5
	.byte	0x23
	.value	0x1d1
	.long	0x21c6
	.uleb128 0x1b
	.long	.LASF335
	.byte	0x23
	.value	0x1ea
	.long	.LASF792
	.long	0x52c0
	.long	0x5303
	.uleb128 0xc
	.long	0x141ed
	.uleb128 0xc
	.long	0x52d8
	.byte	0
	.uleb128 0x1b
	.long	.LASF335
	.byte	0x23
	.value	0x1f8
	.long	.LASF793
	.long	0x52c0
	.long	0x5327
	.uleb128 0xc
	.long	0x141ed
	.uleb128 0xc
	.long	0x52d8
	.uleb128 0xc
	.long	0x52cc
	.byte	0
	.uleb128 0x1a
	.long	.LASF338
	.byte	0x23
	.value	0x204
	.long	.LASF794
	.long	0x5347
	.uleb128 0xc
	.long	0x141ed
	.uleb128 0xc
	.long	0x52c0
	.uleb128 0xc
	.long	0x52d8
	.byte	0
	.uleb128 0x1b
	.long	.LASF119
	.byte	0x23
	.value	0x226
	.long	.LASF795
	.long	0x52d8
	.long	0x5361
	.uleb128 0xc
	.long	0x141f3
	.byte	0
	.uleb128 0x14
	.long	0x52a8
	.uleb128 0x1b
	.long	.LASF341
	.byte	0x23
	.value	0x22f
	.long	.LASF796
	.long	0x52a8
	.long	0x5380
	.uleb128 0xc
	.long	0x141f3
	.byte	0
	.uleb128 0x37
	.long	.LASF343
	.byte	0x23
	.value	0x1dd
	.long	0x5396
	.uleb128 0x20
	.long	.LASF261
	.long	0x5396
	.byte	0
	.uleb128 0x6
	.long	.LASF797
	.byte	0x1
	.byte	0x21
	.byte	0x5c
	.long	0x53fe
	.uleb128 0x34
	.long	0x8aeb
	.byte	0
	.byte	0x1
	.uleb128 0x26
	.long	.LASF328
	.byte	0x21
	.byte	0x71
	.long	.LASF798
	.byte	0x1
	.long	0x53bd
	.long	0x53c3
	.uleb128 0xb
	.long	0x14223
	.byte	0
	.uleb128 0x26
	.long	.LASF328
	.byte	0x21
	.byte	0x73
	.long	.LASF799
	.byte	0x1
	.long	0x53d7
	.long	0x53e2
	.uleb128 0xb
	.long	0x14223
	.uleb128 0xc
	.long	0x14205
	.byte	0
	.uleb128 0x35
	.long	.LASF331
	.byte	0x21
	.byte	0x79
	.long	.LASF800
	.byte	0x1
	.long	0x53f2
	.uleb128 0xb
	.long	0x14223
	.uleb128 0xb
	.long	0x30
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0x5396
	.uleb128 0x7
	.long	.LASF801
	.byte	0x18
	.byte	0xf
	.byte	0x48
	.long	0x56cb
	.uleb128 0x7
	.long	.LASF496
	.byte	0x18
	.byte	0xf
	.byte	0x4f
	.long	0x54d1
	.uleb128 0x8
	.long	0x5396
	.byte	0
	.uleb128 0x9
	.long	.LASF497
	.byte	0xf
	.byte	0x52
	.long	0x54d1
	.byte	0
	.uleb128 0x9
	.long	.LASF498
	.byte	0xf
	.byte	0x53
	.long	0x54d1
	.byte	0x8
	.uleb128 0x9
	.long	.LASF499
	.byte	0xf
	.byte	0x54
	.long	0x54d1
	.byte	0x10
	.uleb128 0xa
	.long	.LASF496
	.byte	0xf
	.byte	0x56
	.long	.LASF802
	.long	0x5458
	.long	0x545e
	.uleb128 0xb
	.long	0x14229
	.byte	0
	.uleb128 0xa
	.long	.LASF496
	.byte	0xf
	.byte	0x5a
	.long	.LASF803
	.long	0x5471
	.long	0x547c
	.uleb128 0xb
	.long	0x14229
	.uleb128 0xc
	.long	0x1422f
	.byte	0
	.uleb128 0xa
	.long	.LASF496
	.byte	0xf
	.byte	0x5f
	.long	.LASF804
	.long	0x548f
	.long	0x549a
	.uleb128 0xb
	.long	0x14229
	.uleb128 0xc
	.long	0x14235
	.byte	0
	.uleb128 0xa
	.long	.LASF503
	.byte	0xf
	.byte	0x65
	.long	.LASF805
	.long	0x54ad
	.long	0x54b8
	.uleb128 0xb
	.long	0x14229
	.uleb128 0xc
	.long	0x1423b
	.byte	0
	.uleb128 0xd
	.long	.LASF696
	.long	.LASF806
	.long	0x54c5
	.uleb128 0xb
	.long	0x14229
	.uleb128 0xb
	.long	0x30
	.byte	0
	.byte	0
	.uleb128 0x16
	.long	.LASF4
	.byte	0xf
	.byte	0x4d
	.long	0x8a0d
	.uleb128 0x16
	.long	.LASF505
	.byte	0xf
	.byte	0x4b
	.long	0x8acc
	.uleb128 0x14
	.long	0x54dc
	.uleb128 0x9
	.long	.LASF506
	.byte	0xf
	.byte	0xa4
	.long	0x540f
	.byte	0
	.uleb128 0x16
	.long	.LASF9
	.byte	0xf
	.byte	0x6e
	.long	0x5396
	.uleb128 0x17
	.long	.LASF507
	.byte	0xf
	.byte	0x71
	.long	.LASF807
	.long	0x14241
	.long	0x551a
	.long	0x5520
	.uleb128 0xb
	.long	0x14247
	.byte	0
	.uleb128 0x17
	.long	.LASF507
	.byte	0xf
	.byte	0x75
	.long	.LASF808
	.long	0x1422f
	.long	0x5537
	.long	0x553d
	.uleb128 0xb
	.long	0x1424d
	.byte	0
	.uleb128 0x17
	.long	.LASF211
	.byte	0xf
	.byte	0x79
	.long	.LASF809
	.long	0x54f8
	.long	0x5554
	.long	0x555a
	.uleb128 0xb
	.long	0x1424d
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x7c
	.long	.LASF810
	.long	0x556d
	.long	0x5573
	.uleb128 0xb
	.long	0x14247
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x7f
	.long	.LASF811
	.long	0x5586
	.long	0x5591
	.uleb128 0xb
	.long	0x14247
	.uleb128 0xc
	.long	0x14253
	.byte	0
	.uleb128 0x14
	.long	0x54f8
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x82
	.long	.LASF812
	.long	0x55a9
	.long	0x55b4
	.uleb128 0xb
	.long	0x14247
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x86
	.long	.LASF813
	.long	0x55c7
	.long	0x55d7
	.uleb128 0xb
	.long	0x14247
	.uleb128 0xc
	.long	0x21c6
	.uleb128 0xc
	.long	0x14253
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x8b
	.long	.LASF814
	.long	0x55ea
	.long	0x55f5
	.uleb128 0xb
	.long	0x14247
	.uleb128 0xc
	.long	0x14235
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x8e
	.long	.LASF815
	.long	0x5608
	.long	0x5613
	.uleb128 0xb
	.long	0x14247
	.uleb128 0xc
	.long	0x14259
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x92
	.long	.LASF816
	.long	0x5626
	.long	0x5636
	.uleb128 0xb
	.long	0x14247
	.uleb128 0xc
	.long	0x14259
	.uleb128 0xc
	.long	0x14253
	.byte	0
	.uleb128 0xa
	.long	.LASF519
	.byte	0xf
	.byte	0x9f
	.long	.LASF817
	.long	0x5649
	.long	0x5654
	.uleb128 0xb
	.long	0x14247
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x17
	.long	.LASF521
	.byte	0xf
	.byte	0xa7
	.long	.LASF818
	.long	0x54d1
	.long	0x566b
	.long	0x5676
	.uleb128 0xb
	.long	0x14247
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0xa
	.long	.LASF523
	.byte	0xf
	.byte	0xae
	.long	.LASF819
	.long	0x5689
	.long	0x5699
	.uleb128 0xb
	.long	0x14247
	.uleb128 0xc
	.long	0x54d1
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0x26
	.long	.LASF525
	.byte	0xf
	.byte	0xb7
	.long	.LASF820
	.byte	0x3
	.long	0x56ad
	.long	0x56b8
	.uleb128 0xb
	.long	0x14247
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x30
	.uleb128 0x20
	.long	.LASF261
	.long	0x5396
	.byte	0
	.uleb128 0x14
	.long	0x5403
	.uleb128 0x6
	.long	.LASF821
	.byte	0x18
	.byte	0xf
	.byte	0xd6
	.long	0x60b3
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x5654
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x5676
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x54ec
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x5520
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x553d
	.uleb128 0x34
	.long	0x5403
	.byte	0
	.byte	0x2
	.uleb128 0xe
	.long	.LASF290
	.byte	0xf
	.byte	0xe2
	.long	0x30
	.byte	0x1
	.uleb128 0xe
	.long	.LASF4
	.byte	0xf
	.byte	0xe3
	.long	0x54d1
	.byte	0x1
	.uleb128 0xe
	.long	.LASF10
	.byte	0xf
	.byte	0xe5
	.long	0x8a18
	.byte	0x1
	.uleb128 0xe
	.long	.LASF11
	.byte	0xf
	.byte	0xe6
	.long	0x8a23
	.byte	0x1
	.uleb128 0xe
	.long	.LASF13
	.byte	0xf
	.byte	0xe7
	.long	0x8c3d
	.byte	0x1
	.uleb128 0xe
	.long	.LASF14
	.byte	0xf
	.byte	0xe9
	.long	0x8c42
	.byte	0x1
	.uleb128 0xe
	.long	.LASF15
	.byte	0xf
	.byte	0xea
	.long	0x60b3
	.byte	0x1
	.uleb128 0xe
	.long	.LASF16
	.byte	0xf
	.byte	0xeb
	.long	0x60b8
	.byte	0x1
	.uleb128 0xe
	.long	.LASF5
	.byte	0xf
	.byte	0xec
	.long	0x21c6
	.byte	0x1
	.uleb128 0xe
	.long	.LASF9
	.byte	0xf
	.byte	0xee
	.long	0x5396
	.byte	0x1
	.uleb128 0x26
	.long	.LASF528
	.byte	0xf
	.byte	0xfd
	.long	.LASF822
	.byte	0x1
	.long	0x5792
	.long	0x5798
	.uleb128 0xb
	.long	0x1425f
	.byte	0
	.uleb128 0x1d
	.long	.LASF528
	.byte	0xf
	.value	0x108
	.long	.LASF823
	.byte	0x1
	.long	0x57ad
	.long	0x57b8
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x14265
	.byte	0
	.uleb128 0x14
	.long	0x5772
	.uleb128 0x1d
	.long	.LASF528
	.byte	0xf
	.value	0x115
	.long	.LASF824
	.byte	0x1
	.long	0x57d2
	.long	0x57e2
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5766
	.uleb128 0xc
	.long	0x14265
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x121
	.long	.LASF825
	.byte	0x1
	.long	0x57f7
	.long	0x580c
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5766
	.uleb128 0xc
	.long	0x1426b
	.uleb128 0xc
	.long	0x14265
	.byte	0
	.uleb128 0x14
	.long	0x5706
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x13e
	.long	.LASF826
	.byte	0x1
	.long	0x5826
	.long	0x5831
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x14271
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x14f
	.long	.LASF827
	.byte	0x1
	.long	0x5846
	.long	0x5851
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x14277
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x153
	.long	.LASF828
	.byte	0x1
	.long	0x5866
	.long	0x5876
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x14271
	.uleb128 0xc
	.long	0x14265
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x15c
	.long	.LASF829
	.byte	0x1
	.long	0x588b
	.long	0x589b
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x14277
	.uleb128 0xc
	.long	0x14265
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x175
	.long	.LASF830
	.byte	0x1
	.long	0x58b0
	.long	0x58c0
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x60c2
	.uleb128 0xc
	.long	0x14265
	.byte	0
	.uleb128 0x1c
	.long	.LASF538
	.byte	0xf
	.value	0x1a7
	.long	.LASF831
	.byte	0x1
	.long	0x58d5
	.long	0x58e0
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x27
	.long	.LASF89
	.byte	0x2f
	.byte	0xa7
	.long	.LASF832
	.long	0x1427d
	.byte	0x1
	.long	0x58f8
	.long	0x5903
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x14271
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0xf
	.value	0x1c0
	.long	.LASF833
	.long	0x1427d
	.byte	0x1
	.long	0x591c
	.long	0x5927
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x14277
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0xf
	.value	0x1d6
	.long	.LASF834
	.long	0x1427d
	.byte	0x1
	.long	0x5940
	.long	0x594b
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x60c2
	.byte	0
	.uleb128 0x1c
	.long	.LASF159
	.byte	0xf
	.value	0x1e8
	.long	.LASF835
	.byte	0x1
	.long	0x5960
	.long	0x5970
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5766
	.uleb128 0xc
	.long	0x1426b
	.byte	0
	.uleb128 0x1c
	.long	.LASF159
	.byte	0xf
	.value	0x215
	.long	.LASF836
	.byte	0x1
	.long	0x5985
	.long	0x5990
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x60c2
	.byte	0
	.uleb128 0x1e
	.long	.LASF96
	.byte	0xf
	.value	0x223
	.long	.LASF837
	.long	0x5736
	.byte	0x1
	.long	0x59a9
	.long	0x59af
	.uleb128 0xb
	.long	0x1425f
	.byte	0
	.uleb128 0x1e
	.long	.LASF96
	.byte	0xf
	.value	0x22c
	.long	.LASF838
	.long	0x5742
	.byte	0x1
	.long	0x59c8
	.long	0x59ce
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x1f
	.string	"end"
	.byte	0xf
	.value	0x235
	.long	.LASF839
	.long	0x5736
	.byte	0x1
	.long	0x59e7
	.long	0x59ed
	.uleb128 0xb
	.long	0x1425f
	.byte	0
	.uleb128 0x1f
	.string	"end"
	.byte	0xf
	.value	0x23e
	.long	.LASF840
	.long	0x5742
	.byte	0x1
	.long	0x5a06
	.long	0x5a0c
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x1e
	.long	.LASF101
	.byte	0xf
	.value	0x247
	.long	.LASF841
	.long	0x575a
	.byte	0x1
	.long	0x5a25
	.long	0x5a2b
	.uleb128 0xb
	.long	0x1425f
	.byte	0
	.uleb128 0x1e
	.long	.LASF101
	.byte	0xf
	.value	0x250
	.long	.LASF842
	.long	0x574e
	.byte	0x1
	.long	0x5a44
	.long	0x5a4a
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x1e
	.long	.LASF104
	.byte	0xf
	.value	0x259
	.long	.LASF843
	.long	0x575a
	.byte	0x1
	.long	0x5a63
	.long	0x5a69
	.uleb128 0xb
	.long	0x1425f
	.byte	0
	.uleb128 0x1e
	.long	.LASF104
	.byte	0xf
	.value	0x262
	.long	.LASF844
	.long	0x574e
	.byte	0x1
	.long	0x5a82
	.long	0x5a88
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x1e
	.long	.LASF107
	.byte	0xf
	.value	0x26c
	.long	.LASF845
	.long	0x5742
	.byte	0x1
	.long	0x5aa1
	.long	0x5aa7
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x1e
	.long	.LASF109
	.byte	0xf
	.value	0x275
	.long	.LASF846
	.long	0x5742
	.byte	0x1
	.long	0x5ac0
	.long	0x5ac6
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x1e
	.long	.LASF111
	.byte	0xf
	.value	0x27e
	.long	.LASF847
	.long	0x574e
	.byte	0x1
	.long	0x5adf
	.long	0x5ae5
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x1e
	.long	.LASF113
	.byte	0xf
	.value	0x287
	.long	.LASF848
	.long	0x574e
	.byte	0x1
	.long	0x5afe
	.long	0x5b04
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x1e
	.long	.LASF115
	.byte	0xf
	.value	0x28e
	.long	.LASF849
	.long	0x5766
	.byte	0x1
	.long	0x5b1d
	.long	0x5b23
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x1e
	.long	.LASF119
	.byte	0xf
	.value	0x293
	.long	.LASF850
	.long	0x5766
	.byte	0x1
	.long	0x5b3c
	.long	0x5b42
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x1c
	.long	.LASF121
	.byte	0xf
	.value	0x2a1
	.long	.LASF851
	.byte	0x1
	.long	0x5b57
	.long	0x5b62
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5766
	.byte	0
	.uleb128 0x1c
	.long	.LASF121
	.byte	0xf
	.value	0x2b5
	.long	.LASF852
	.byte	0x1
	.long	0x5b77
	.long	0x5b87
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5766
	.uleb128 0xc
	.long	0x1426b
	.byte	0
	.uleb128 0x1c
	.long	.LASF124
	.byte	0xf
	.value	0x2d5
	.long	.LASF853
	.byte	0x1
	.long	0x5b9c
	.long	0x5ba2
	.uleb128 0xb
	.long	0x1425f
	.byte	0
	.uleb128 0x1e
	.long	.LASF126
	.byte	0xf
	.value	0x2de
	.long	.LASF854
	.long	0x5766
	.byte	0x1
	.long	0x5bbb
	.long	0x5bc1
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x1e
	.long	.LASF132
	.byte	0xf
	.value	0x2e7
	.long	.LASF855
	.long	0x9ef1
	.byte	0x1
	.long	0x5bda
	.long	0x5be0
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x26
	.long	.LASF128
	.byte	0x2f
	.byte	0x41
	.long	.LASF856
	.byte	0x1
	.long	0x5bf4
	.long	0x5bff
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5766
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0xf
	.value	0x30b
	.long	.LASF857
	.long	0x571e
	.byte	0x1
	.long	0x5c18
	.long	0x5c23
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5766
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0xf
	.value	0x31a
	.long	.LASF858
	.long	0x572a
	.byte	0x1
	.long	0x5c3c
	.long	0x5c47
	.uleb128 0xb
	.long	0x14283
	.uleb128 0xc
	.long	0x5766
	.byte	0
	.uleb128 0x1c
	.long	.LASF567
	.byte	0xf
	.value	0x320
	.long	.LASF859
	.byte	0x2
	.long	0x5c5c
	.long	0x5c67
	.uleb128 0xb
	.long	0x14283
	.uleb128 0xc
	.long	0x5766
	.byte	0
	.uleb128 0x1f
	.string	"at"
	.byte	0xf
	.value	0x336
	.long	.LASF860
	.long	0x571e
	.byte	0x1
	.long	0x5c7f
	.long	0x5c8a
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5766
	.byte	0
	.uleb128 0x1f
	.string	"at"
	.byte	0xf
	.value	0x348
	.long	.LASF861
	.long	0x572a
	.byte	0x1
	.long	0x5ca2
	.long	0x5cad
	.uleb128 0xb
	.long	0x14283
	.uleb128 0xc
	.long	0x5766
	.byte	0
	.uleb128 0x1e
	.long	.LASF139
	.byte	0xf
	.value	0x353
	.long	.LASF862
	.long	0x571e
	.byte	0x1
	.long	0x5cc6
	.long	0x5ccc
	.uleb128 0xb
	.long	0x1425f
	.byte	0
	.uleb128 0x1e
	.long	.LASF139
	.byte	0xf
	.value	0x35b
	.long	.LASF863
	.long	0x572a
	.byte	0x1
	.long	0x5ce5
	.long	0x5ceb
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x1e
	.long	.LASF142
	.byte	0xf
	.value	0x363
	.long	.LASF864
	.long	0x571e
	.byte	0x1
	.long	0x5d04
	.long	0x5d0a
	.uleb128 0xb
	.long	0x1425f
	.byte	0
	.uleb128 0x1e
	.long	.LASF142
	.byte	0xf
	.value	0x36b
	.long	.LASF865
	.long	0x572a
	.byte	0x1
	.long	0x5d23
	.long	0x5d29
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x1e
	.long	.LASF209
	.byte	0xf
	.value	0x37a
	.long	.LASF866
	.long	0xaa8a
	.byte	0x1
	.long	0x5d42
	.long	0x5d48
	.uleb128 0xb
	.long	0x1425f
	.byte	0
	.uleb128 0x1e
	.long	.LASF209
	.byte	0xf
	.value	0x382
	.long	.LASF867
	.long	0x9720
	.byte	0x1
	.long	0x5d61
	.long	0x5d67
	.uleb128 0xb
	.long	0x14283
	.byte	0
	.uleb128 0x1c
	.long	.LASF157
	.byte	0xf
	.value	0x391
	.long	.LASF868
	.byte	0x1
	.long	0x5d7c
	.long	0x5d87
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x1426b
	.byte	0
	.uleb128 0x1c
	.long	.LASF157
	.byte	0xf
	.value	0x3a3
	.long	.LASF869
	.byte	0x1
	.long	0x5d9c
	.long	0x5da7
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x14289
	.byte	0
	.uleb128 0x1c
	.long	.LASF180
	.byte	0xf
	.value	0x3b5
	.long	.LASF870
	.byte	0x1
	.long	0x5dbc
	.long	0x5dc2
	.uleb128 0xb
	.long	0x1425f
	.byte	0
	.uleb128 0x27
	.long	.LASF167
	.byte	0x2f
	.byte	0x6b
	.long	.LASF871
	.long	0x5736
	.byte	0x1
	.long	0x5dda
	.long	0x5dea
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5742
	.uleb128 0xc
	.long	0x1426b
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0xf
	.value	0x3f6
	.long	.LASF872
	.long	0x5736
	.byte	0x1
	.long	0x5e03
	.long	0x5e13
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5742
	.uleb128 0xc
	.long	0x14289
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0xf
	.value	0x407
	.long	.LASF873
	.long	0x5736
	.byte	0x1
	.long	0x5e2c
	.long	0x5e3c
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5742
	.uleb128 0xc
	.long	0x60c2
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0xf
	.value	0x41b
	.long	.LASF874
	.long	0x5736
	.byte	0x1
	.long	0x5e55
	.long	0x5e6a
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5742
	.uleb128 0xc
	.long	0x5766
	.uleb128 0xc
	.long	0x1426b
	.byte	0
	.uleb128 0x1e
	.long	.LASF176
	.byte	0xf
	.value	0x47a
	.long	.LASF875
	.long	0x5736
	.byte	0x1
	.long	0x5e83
	.long	0x5e8e
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5742
	.byte	0
	.uleb128 0x1e
	.long	.LASF176
	.byte	0xf
	.value	0x495
	.long	.LASF876
	.long	0x5736
	.byte	0x1
	.long	0x5ea7
	.long	0x5eb7
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5742
	.uleb128 0xc
	.long	0x5742
	.byte	0
	.uleb128 0x1c
	.long	.LASF205
	.byte	0xf
	.value	0x4aa
	.long	.LASF877
	.byte	0x1
	.long	0x5ecc
	.long	0x5ed7
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x1427d
	.byte	0
	.uleb128 0x1c
	.long	.LASF130
	.byte	0xf
	.value	0x4bb
	.long	.LASF878
	.byte	0x1
	.long	0x5eec
	.long	0x5ef2
	.uleb128 0xb
	.long	0x1425f
	.byte	0
	.uleb128 0x1c
	.long	.LASF588
	.byte	0xf
	.value	0x512
	.long	.LASF879
	.byte	0x2
	.long	0x5f07
	.long	0x5f17
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5766
	.uleb128 0xc
	.long	0x1426b
	.byte	0
	.uleb128 0x1c
	.long	.LASF590
	.byte	0xf
	.value	0x51c
	.long	.LASF880
	.byte	0x2
	.long	0x5f2c
	.long	0x5f37
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5766
	.byte	0
	.uleb128 0x26
	.long	.LASF592
	.byte	0x2f
	.byte	0xe1
	.long	.LASF881
	.byte	0x2
	.long	0x5f4b
	.long	0x5f5b
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x21c6
	.uleb128 0xc
	.long	0x1426b
	.byte	0
	.uleb128 0x1c
	.long	.LASF594
	.byte	0x2f
	.value	0x1c1
	.long	.LASF882
	.byte	0x2
	.long	0x5f70
	.long	0x5f85
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5736
	.uleb128 0xc
	.long	0x5766
	.uleb128 0xc
	.long	0x1426b
	.byte	0
	.uleb128 0x1c
	.long	.LASF596
	.byte	0x2f
	.value	0x21c
	.long	.LASF883
	.byte	0x2
	.long	0x5f9a
	.long	0x5fa5
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5766
	.byte	0
	.uleb128 0x1e
	.long	.LASF598
	.byte	0x2f
	.value	0x24e
	.long	.LASF884
	.long	0x9ef1
	.byte	0x2
	.long	0x5fbe
	.long	0x5fc4
	.uleb128 0xb
	.long	0x1425f
	.byte	0
	.uleb128 0x1e
	.long	.LASF600
	.byte	0xf
	.value	0x58e
	.long	.LASF885
	.long	0x5766
	.byte	0x2
	.long	0x5fdd
	.long	0x5fed
	.uleb128 0xb
	.long	0x14283
	.uleb128 0xc
	.long	0x5766
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1c
	.long	.LASF602
	.byte	0xf
	.value	0x59c
	.long	.LASF886
	.byte	0x2
	.long	0x6002
	.long	0x600d
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5712
	.byte	0
	.uleb128 0x27
	.long	.LASF73
	.byte	0x2f
	.byte	0x8d
	.long	.LASF887
	.long	0x5736
	.byte	0x2
	.long	0x6025
	.long	0x6030
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5736
	.byte	0
	.uleb128 0x27
	.long	.LASF73
	.byte	0x2f
	.byte	0x99
	.long	.LASF888
	.long	0x5736
	.byte	0x2
	.long	0x6048
	.long	0x6058
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x5736
	.uleb128 0xc
	.long	0x5736
	.byte	0
	.uleb128 0x19
	.long	.LASF606
	.byte	0xf
	.value	0x5ae
	.long	.LASF889
	.long	0x606c
	.long	0x607c
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x14277
	.uleb128 0xc
	.long	0x22bd
	.byte	0
	.uleb128 0x19
	.long	.LASF606
	.byte	0xf
	.value	0x5b9
	.long	.LASF890
	.long	0x6090
	.long	0x60a0
	.uleb128 0xb
	.long	0x1425f
	.uleb128 0xc
	.long	0x14277
	.uleb128 0xc
	.long	0x2523
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x30
	.uleb128 0x21
	.long	.LASF261
	.long	0x5396
	.byte	0
	.uleb128 0x2a
	.long	.LASF891
	.uleb128 0x2a
	.long	.LASF892
	.uleb128 0x14
	.long	0x56d0
	.uleb128 0x6
	.long	.LASF893
	.byte	0x10
	.byte	0x24
	.byte	0x2f
	.long	0x61aa
	.uleb128 0xe
	.long	.LASF13
	.byte	0x24
	.byte	0x36
	.long	0x9720
	.byte	0x1
	.uleb128 0x9
	.long	.LASF349
	.byte	0x24
	.byte	0x3a
	.long	0x60ce
	.byte	0
	.uleb128 0xe
	.long	.LASF5
	.byte	0x24
	.byte	0x35
	.long	0x21c6
	.byte	0x1
	.uleb128 0x9
	.long	.LASF350
	.byte	0x24
	.byte	0x3b
	.long	0x60e6
	.byte	0x8
	.uleb128 0xe
	.long	.LASF14
	.byte	0x24
	.byte	0x37
	.long	0x9720
	.byte	0x1
	.uleb128 0xa
	.long	.LASF351
	.byte	0x24
	.byte	0x3e
	.long	.LASF894
	.long	0x611d
	.long	0x612d
	.uleb128 0xb
	.long	0x14488
	.uleb128 0xc
	.long	0x60fe
	.uleb128 0xc
	.long	0x60e6
	.byte	0
	.uleb128 0x26
	.long	.LASF351
	.byte	0x24
	.byte	0x42
	.long	.LASF895
	.byte	0x1
	.long	0x6141
	.long	0x6147
	.uleb128 0xb
	.long	0x14488
	.byte	0
	.uleb128 0x27
	.long	.LASF115
	.byte	0x24
	.byte	0x47
	.long	.LASF896
	.long	0x60e6
	.byte	0x1
	.long	0x615f
	.long	0x6165
	.uleb128 0xb
	.long	0x1448e
	.byte	0
	.uleb128 0x27
	.long	.LASF96
	.byte	0x24
	.byte	0x4b
	.long	.LASF897
	.long	0x60fe
	.byte	0x1
	.long	0x617d
	.long	0x6183
	.uleb128 0xb
	.long	0x1448e
	.byte	0
	.uleb128 0x38
	.string	"end"
	.byte	0x24
	.byte	0x4f
	.long	.LASF898
	.long	0x60fe
	.byte	0x1
	.long	0x619b
	.long	0x61a1
	.uleb128 0xb
	.long	0x1448e
	.byte	0
	.uleb128 0x2c
	.string	"_E"
	.long	0x30
	.byte	0
	.uleb128 0x6
	.long	.LASF899
	.byte	0x18
	.byte	0xf
	.byte	0xd6
	.long	0x6b8d
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x6f4b
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x6f6d
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x6de3
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x6e17
	.uleb128 0x23
	.byte	0xf
	.byte	0xd6
	.long	0x6e34
	.uleb128 0x34
	.long	0x6cfa
	.byte	0
	.byte	0x2
	.uleb128 0xe
	.long	.LASF290
	.byte	0xf
	.byte	0xe2
	.long	0xc3f6
	.byte	0x1
	.uleb128 0xe
	.long	.LASF4
	.byte	0xf
	.byte	0xe3
	.long	0x6dc8
	.byte	0x1
	.uleb128 0xe
	.long	.LASF10
	.byte	0xf
	.byte	0xe5
	.long	0x8ea6
	.byte	0x1
	.uleb128 0xe
	.long	.LASF11
	.byte	0xf
	.byte	0xe6
	.long	0x8eb1
	.byte	0x1
	.uleb128 0xe
	.long	.LASF13
	.byte	0xf
	.byte	0xe7
	.long	0x90cb
	.byte	0x1
	.uleb128 0xe
	.long	.LASF14
	.byte	0xf
	.byte	0xe9
	.long	0x90d0
	.byte	0x1
	.uleb128 0xe
	.long	.LASF15
	.byte	0xf
	.byte	0xea
	.long	0x6fc7
	.byte	0x1
	.uleb128 0xe
	.long	.LASF16
	.byte	0xf
	.byte	0xeb
	.long	0x6fcc
	.byte	0x1
	.uleb128 0xe
	.long	.LASF5
	.byte	0xf
	.byte	0xec
	.long	0x21c6
	.byte	0x1
	.uleb128 0xe
	.long	.LASF9
	.byte	0xf
	.byte	0xee
	.long	0x6c8d
	.byte	0x1
	.uleb128 0x26
	.long	.LASF528
	.byte	0xf
	.byte	0xfd
	.long	.LASF900
	.byte	0x1
	.long	0x626c
	.long	0x6272
	.uleb128 0xb
	.long	0x1445d
	.byte	0
	.uleb128 0x1d
	.long	.LASF528
	.byte	0xf
	.value	0x108
	.long	.LASF901
	.byte	0x1
	.long	0x6287
	.long	0x6292
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x14463
	.byte	0
	.uleb128 0x14
	.long	0x624c
	.uleb128 0x1d
	.long	.LASF528
	.byte	0xf
	.value	0x115
	.long	.LASF902
	.byte	0x1
	.long	0x62ac
	.long	0x62bc
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6240
	.uleb128 0xc
	.long	0x14463
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x121
	.long	.LASF903
	.byte	0x1
	.long	0x62d1
	.long	0x62e6
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6240
	.uleb128 0xc
	.long	0x14469
	.uleb128 0xc
	.long	0x14463
	.byte	0
	.uleb128 0x14
	.long	0x61e0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x13e
	.long	.LASF904
	.byte	0x1
	.long	0x6300
	.long	0x630b
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x142e0
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x14f
	.long	.LASF905
	.byte	0x1
	.long	0x6320
	.long	0x632b
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x1446f
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x153
	.long	.LASF906
	.byte	0x1
	.long	0x6340
	.long	0x6350
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x142e0
	.uleb128 0xc
	.long	0x14463
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x15c
	.long	.LASF907
	.byte	0x1
	.long	0x6365
	.long	0x6375
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x1446f
	.uleb128 0xc
	.long	0x14463
	.byte	0
	.uleb128 0x1c
	.long	.LASF528
	.byte	0xf
	.value	0x175
	.long	.LASF908
	.byte	0x1
	.long	0x638a
	.long	0x639a
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6fd1
	.uleb128 0xc
	.long	0x14463
	.byte	0
	.uleb128 0x1c
	.long	.LASF538
	.byte	0xf
	.value	0x1a7
	.long	.LASF909
	.byte	0x1
	.long	0x63af
	.long	0x63ba
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x27
	.long	.LASF89
	.byte	0x2f
	.byte	0xa7
	.long	.LASF910
	.long	0x142da
	.byte	0x1
	.long	0x63d2
	.long	0x63dd
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x142e0
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0xf
	.value	0x1c0
	.long	.LASF911
	.long	0x142da
	.byte	0x1
	.long	0x63f6
	.long	0x6401
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x1446f
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0xf
	.value	0x1d6
	.long	.LASF912
	.long	0x142da
	.byte	0x1
	.long	0x641a
	.long	0x6425
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6fd1
	.byte	0
	.uleb128 0x1c
	.long	.LASF159
	.byte	0xf
	.value	0x1e8
	.long	.LASF913
	.byte	0x1
	.long	0x643a
	.long	0x644a
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6240
	.uleb128 0xc
	.long	0x14469
	.byte	0
	.uleb128 0x1c
	.long	.LASF159
	.byte	0xf
	.value	0x215
	.long	.LASF914
	.byte	0x1
	.long	0x645f
	.long	0x646a
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6fd1
	.byte	0
	.uleb128 0x1e
	.long	.LASF96
	.byte	0xf
	.value	0x223
	.long	.LASF915
	.long	0x6210
	.byte	0x1
	.long	0x6483
	.long	0x6489
	.uleb128 0xb
	.long	0x1445d
	.byte	0
	.uleb128 0x1e
	.long	.LASF96
	.byte	0xf
	.value	0x22c
	.long	.LASF916
	.long	0x621c
	.byte	0x1
	.long	0x64a2
	.long	0x64a8
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x1f
	.string	"end"
	.byte	0xf
	.value	0x235
	.long	.LASF917
	.long	0x6210
	.byte	0x1
	.long	0x64c1
	.long	0x64c7
	.uleb128 0xb
	.long	0x1445d
	.byte	0
	.uleb128 0x1f
	.string	"end"
	.byte	0xf
	.value	0x23e
	.long	.LASF918
	.long	0x621c
	.byte	0x1
	.long	0x64e0
	.long	0x64e6
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x1e
	.long	.LASF101
	.byte	0xf
	.value	0x247
	.long	.LASF919
	.long	0x6234
	.byte	0x1
	.long	0x64ff
	.long	0x6505
	.uleb128 0xb
	.long	0x1445d
	.byte	0
	.uleb128 0x1e
	.long	.LASF101
	.byte	0xf
	.value	0x250
	.long	.LASF920
	.long	0x6228
	.byte	0x1
	.long	0x651e
	.long	0x6524
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x1e
	.long	.LASF104
	.byte	0xf
	.value	0x259
	.long	.LASF921
	.long	0x6234
	.byte	0x1
	.long	0x653d
	.long	0x6543
	.uleb128 0xb
	.long	0x1445d
	.byte	0
	.uleb128 0x1e
	.long	.LASF104
	.byte	0xf
	.value	0x262
	.long	.LASF922
	.long	0x6228
	.byte	0x1
	.long	0x655c
	.long	0x6562
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x1e
	.long	.LASF107
	.byte	0xf
	.value	0x26c
	.long	.LASF923
	.long	0x621c
	.byte	0x1
	.long	0x657b
	.long	0x6581
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x1e
	.long	.LASF109
	.byte	0xf
	.value	0x275
	.long	.LASF924
	.long	0x621c
	.byte	0x1
	.long	0x659a
	.long	0x65a0
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x1e
	.long	.LASF111
	.byte	0xf
	.value	0x27e
	.long	.LASF925
	.long	0x6228
	.byte	0x1
	.long	0x65b9
	.long	0x65bf
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x1e
	.long	.LASF113
	.byte	0xf
	.value	0x287
	.long	.LASF926
	.long	0x6228
	.byte	0x1
	.long	0x65d8
	.long	0x65de
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x1e
	.long	.LASF115
	.byte	0xf
	.value	0x28e
	.long	.LASF927
	.long	0x6240
	.byte	0x1
	.long	0x65f7
	.long	0x65fd
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x1e
	.long	.LASF119
	.byte	0xf
	.value	0x293
	.long	.LASF928
	.long	0x6240
	.byte	0x1
	.long	0x6616
	.long	0x661c
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x1c
	.long	.LASF121
	.byte	0xf
	.value	0x2a1
	.long	.LASF929
	.byte	0x1
	.long	0x6631
	.long	0x663c
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6240
	.byte	0
	.uleb128 0x1c
	.long	.LASF121
	.byte	0xf
	.value	0x2b5
	.long	.LASF930
	.byte	0x1
	.long	0x6651
	.long	0x6661
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6240
	.uleb128 0xc
	.long	0x14469
	.byte	0
	.uleb128 0x1c
	.long	.LASF124
	.byte	0xf
	.value	0x2d5
	.long	.LASF931
	.byte	0x1
	.long	0x6676
	.long	0x667c
	.uleb128 0xb
	.long	0x1445d
	.byte	0
	.uleb128 0x1e
	.long	.LASF126
	.byte	0xf
	.value	0x2de
	.long	.LASF932
	.long	0x6240
	.byte	0x1
	.long	0x6695
	.long	0x669b
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x1e
	.long	.LASF132
	.byte	0xf
	.value	0x2e7
	.long	.LASF933
	.long	0x9ef1
	.byte	0x1
	.long	0x66b4
	.long	0x66ba
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x26
	.long	.LASF128
	.byte	0x2f
	.byte	0x41
	.long	.LASF934
	.byte	0x1
	.long	0x66ce
	.long	0x66d9
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6240
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0xf
	.value	0x30b
	.long	.LASF935
	.long	0x61f8
	.byte	0x1
	.long	0x66f2
	.long	0x66fd
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6240
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0xf
	.value	0x31a
	.long	.LASF936
	.long	0x6204
	.byte	0x1
	.long	0x6716
	.long	0x6721
	.uleb128 0xb
	.long	0x14475
	.uleb128 0xc
	.long	0x6240
	.byte	0
	.uleb128 0x1c
	.long	.LASF567
	.byte	0xf
	.value	0x320
	.long	.LASF937
	.byte	0x2
	.long	0x6736
	.long	0x6741
	.uleb128 0xb
	.long	0x14475
	.uleb128 0xc
	.long	0x6240
	.byte	0
	.uleb128 0x1f
	.string	"at"
	.byte	0xf
	.value	0x336
	.long	.LASF938
	.long	0x61f8
	.byte	0x1
	.long	0x6759
	.long	0x6764
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6240
	.byte	0
	.uleb128 0x1f
	.string	"at"
	.byte	0xf
	.value	0x348
	.long	.LASF939
	.long	0x6204
	.byte	0x1
	.long	0x677c
	.long	0x6787
	.uleb128 0xb
	.long	0x14475
	.uleb128 0xc
	.long	0x6240
	.byte	0
	.uleb128 0x1e
	.long	.LASF139
	.byte	0xf
	.value	0x353
	.long	.LASF940
	.long	0x61f8
	.byte	0x1
	.long	0x67a0
	.long	0x67a6
	.uleb128 0xb
	.long	0x1445d
	.byte	0
	.uleb128 0x1e
	.long	.LASF139
	.byte	0xf
	.value	0x35b
	.long	.LASF941
	.long	0x6204
	.byte	0x1
	.long	0x67bf
	.long	0x67c5
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x1e
	.long	.LASF142
	.byte	0xf
	.value	0x363
	.long	.LASF942
	.long	0x61f8
	.byte	0x1
	.long	0x67de
	.long	0x67e4
	.uleb128 0xb
	.long	0x1445d
	.byte	0
	.uleb128 0x1e
	.long	.LASF142
	.byte	0xf
	.value	0x36b
	.long	.LASF943
	.long	0x6204
	.byte	0x1
	.long	0x67fd
	.long	0x6803
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x1e
	.long	.LASF209
	.byte	0xf
	.value	0x37a
	.long	.LASF944
	.long	0x14013
	.byte	0x1
	.long	0x681c
	.long	0x6822
	.uleb128 0xb
	.long	0x1445d
	.byte	0
	.uleb128 0x1e
	.long	.LASF209
	.byte	0xf
	.value	0x382
	.long	.LASF945
	.long	0x1406a
	.byte	0x1
	.long	0x683b
	.long	0x6841
	.uleb128 0xb
	.long	0x14475
	.byte	0
	.uleb128 0x1c
	.long	.LASF157
	.byte	0xf
	.value	0x391
	.long	.LASF946
	.byte	0x1
	.long	0x6856
	.long	0x6861
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x14469
	.byte	0
	.uleb128 0x1c
	.long	.LASF157
	.byte	0xf
	.value	0x3a3
	.long	.LASF947
	.byte	0x1
	.long	0x6876
	.long	0x6881
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x1447b
	.byte	0
	.uleb128 0x1c
	.long	.LASF180
	.byte	0xf
	.value	0x3b5
	.long	.LASF948
	.byte	0x1
	.long	0x6896
	.long	0x689c
	.uleb128 0xb
	.long	0x1445d
	.byte	0
	.uleb128 0x27
	.long	.LASF167
	.byte	0x2f
	.byte	0x6b
	.long	.LASF949
	.long	0x6210
	.byte	0x1
	.long	0x68b4
	.long	0x68c4
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x621c
	.uleb128 0xc
	.long	0x14469
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0xf
	.value	0x3f6
	.long	.LASF950
	.long	0x6210
	.byte	0x1
	.long	0x68dd
	.long	0x68ed
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x621c
	.uleb128 0xc
	.long	0x1447b
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0xf
	.value	0x407
	.long	.LASF951
	.long	0x6210
	.byte	0x1
	.long	0x6906
	.long	0x6916
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x621c
	.uleb128 0xc
	.long	0x6fd1
	.byte	0
	.uleb128 0x1e
	.long	.LASF167
	.byte	0xf
	.value	0x41b
	.long	.LASF952
	.long	0x6210
	.byte	0x1
	.long	0x692f
	.long	0x6944
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x621c
	.uleb128 0xc
	.long	0x6240
	.uleb128 0xc
	.long	0x14469
	.byte	0
	.uleb128 0x1e
	.long	.LASF176
	.byte	0xf
	.value	0x47a
	.long	.LASF953
	.long	0x6210
	.byte	0x1
	.long	0x695d
	.long	0x6968
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x621c
	.byte	0
	.uleb128 0x1e
	.long	.LASF176
	.byte	0xf
	.value	0x495
	.long	.LASF954
	.long	0x6210
	.byte	0x1
	.long	0x6981
	.long	0x6991
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x621c
	.uleb128 0xc
	.long	0x621c
	.byte	0
	.uleb128 0x1c
	.long	.LASF205
	.byte	0xf
	.value	0x4aa
	.long	.LASF955
	.byte	0x1
	.long	0x69a6
	.long	0x69b1
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x142da
	.byte	0
	.uleb128 0x1c
	.long	.LASF130
	.byte	0xf
	.value	0x4bb
	.long	.LASF956
	.byte	0x1
	.long	0x69c6
	.long	0x69cc
	.uleb128 0xb
	.long	0x1445d
	.byte	0
	.uleb128 0x1c
	.long	.LASF588
	.byte	0xf
	.value	0x512
	.long	.LASF957
	.byte	0x2
	.long	0x69e1
	.long	0x69f1
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6240
	.uleb128 0xc
	.long	0x14469
	.byte	0
	.uleb128 0x1c
	.long	.LASF590
	.byte	0xf
	.value	0x51c
	.long	.LASF958
	.byte	0x2
	.long	0x6a06
	.long	0x6a11
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6240
	.byte	0
	.uleb128 0x26
	.long	.LASF592
	.byte	0x2f
	.byte	0xe1
	.long	.LASF959
	.byte	0x2
	.long	0x6a25
	.long	0x6a35
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x21c6
	.uleb128 0xc
	.long	0x14469
	.byte	0
	.uleb128 0x1c
	.long	.LASF594
	.byte	0x2f
	.value	0x1c1
	.long	.LASF960
	.byte	0x2
	.long	0x6a4a
	.long	0x6a5f
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6210
	.uleb128 0xc
	.long	0x6240
	.uleb128 0xc
	.long	0x14469
	.byte	0
	.uleb128 0x1c
	.long	.LASF596
	.byte	0x2f
	.value	0x21c
	.long	.LASF961
	.byte	0x2
	.long	0x6a74
	.long	0x6a7f
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6240
	.byte	0
	.uleb128 0x1e
	.long	.LASF598
	.byte	0x2f
	.value	0x24e
	.long	.LASF962
	.long	0x9ef1
	.byte	0x2
	.long	0x6a98
	.long	0x6a9e
	.uleb128 0xb
	.long	0x1445d
	.byte	0
	.uleb128 0x1e
	.long	.LASF600
	.byte	0xf
	.value	0x58e
	.long	.LASF963
	.long	0x6240
	.byte	0x2
	.long	0x6ab7
	.long	0x6ac7
	.uleb128 0xb
	.long	0x14475
	.uleb128 0xc
	.long	0x6240
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1c
	.long	.LASF602
	.byte	0xf
	.value	0x59c
	.long	.LASF964
	.byte	0x2
	.long	0x6adc
	.long	0x6ae7
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x61ec
	.byte	0
	.uleb128 0x27
	.long	.LASF73
	.byte	0x2f
	.byte	0x8d
	.long	.LASF965
	.long	0x6210
	.byte	0x2
	.long	0x6aff
	.long	0x6b0a
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6210
	.byte	0
	.uleb128 0x27
	.long	.LASF73
	.byte	0x2f
	.byte	0x99
	.long	.LASF966
	.long	0x6210
	.byte	0x2
	.long	0x6b22
	.long	0x6b32
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x6210
	.uleb128 0xc
	.long	0x6210
	.byte	0
	.uleb128 0x19
	.long	.LASF606
	.byte	0xf
	.value	0x5ae
	.long	.LASF967
	.long	0x6b46
	.long	0x6b56
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x1446f
	.uleb128 0xc
	.long	0x22bd
	.byte	0
	.uleb128 0x19
	.long	.LASF606
	.byte	0xf
	.value	0x5b9
	.long	.LASF968
	.long	0x6b6a
	.long	0x6b7a
	.uleb128 0xb
	.long	0x1445d
	.uleb128 0xc
	.long	0x1446f
	.uleb128 0xc
	.long	0x2523
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0xc3f6
	.uleb128 0x21
	.long	.LASF261
	.long	0x6c8d
	.byte	0
	.uleb128 0x14
	.long	0x61aa
	.uleb128 0x36
	.long	.LASF969
	.byte	0x1
	.byte	0x23
	.value	0x1ba
	.long	0x6c8d
	.uleb128 0x37
	.long	.LASF9
	.byte	0x23
	.value	0x1bd
	.long	0x6c8d
	.uleb128 0x37
	.long	.LASF290
	.byte	0x23
	.value	0x1bf
	.long	0xc3f6
	.uleb128 0x37
	.long	.LASF4
	.byte	0x23
	.value	0x1c2
	.long	0x14013
	.uleb128 0x37
	.long	.LASF334
	.byte	0x23
	.value	0x1cb
	.long	0xa1f5
	.uleb128 0x37
	.long	.LASF5
	.byte	0x23
	.value	0x1d1
	.long	0x21c6
	.uleb128 0x1b
	.long	.LASF335
	.byte	0x23
	.value	0x1ea
	.long	.LASF970
	.long	0x6bb7
	.long	0x6bfa
	.uleb128 0xc
	.long	0x143eb
	.uleb128 0xc
	.long	0x6bcf
	.byte	0
	.uleb128 0x1b
	.long	.LASF335
	.byte	0x23
	.value	0x1f8
	.long	.LASF971
	.long	0x6bb7
	.long	0x6c1e
	.uleb128 0xc
	.long	0x143eb
	.uleb128 0xc
	.long	0x6bcf
	.uleb128 0xc
	.long	0x6bc3
	.byte	0
	.uleb128 0x1a
	.long	.LASF338
	.byte	0x23
	.value	0x204
	.long	.LASF972
	.long	0x6c3e
	.uleb128 0xc
	.long	0x143eb
	.uleb128 0xc
	.long	0x6bb7
	.uleb128 0xc
	.long	0x6bcf
	.byte	0
	.uleb128 0x1b
	.long	.LASF119
	.byte	0x23
	.value	0x226
	.long	.LASF973
	.long	0x6bcf
	.long	0x6c58
	.uleb128 0xc
	.long	0x143f1
	.byte	0
	.uleb128 0x14
	.long	0x6b9f
	.uleb128 0x1b
	.long	.LASF341
	.byte	0x23
	.value	0x22f
	.long	.LASF974
	.long	0x6b9f
	.long	0x6c77
	.uleb128 0xc
	.long	0x143f1
	.byte	0
	.uleb128 0x37
	.long	.LASF343
	.byte	0x23
	.value	0x1dd
	.long	0x6c8d
	.uleb128 0x20
	.long	.LASF261
	.long	0x6c8d
	.byte	0
	.uleb128 0x6
	.long	.LASF975
	.byte	0x1
	.byte	0x21
	.byte	0x5c
	.long	0x6cf5
	.uleb128 0x34
	.long	0x8f79
	.byte	0
	.byte	0x1
	.uleb128 0x26
	.long	.LASF328
	.byte	0x21
	.byte	0x71
	.long	.LASF976
	.byte	0x1
	.long	0x6cb4
	.long	0x6cba
	.uleb128 0xb
	.long	0x14421
	.byte	0
	.uleb128 0x26
	.long	.LASF328
	.byte	0x21
	.byte	0x73
	.long	.LASF977
	.byte	0x1
	.long	0x6cce
	.long	0x6cd9
	.uleb128 0xb
	.long	0x14421
	.uleb128 0xc
	.long	0x14403
	.byte	0
	.uleb128 0x35
	.long	.LASF331
	.byte	0x21
	.byte	0x79
	.long	.LASF978
	.byte	0x1
	.long	0x6ce9
	.uleb128 0xb
	.long	0x14421
	.uleb128 0xb
	.long	0x30
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0x6c8d
	.uleb128 0x7
	.long	.LASF979
	.byte	0x18
	.byte	0xf
	.byte	0x48
	.long	0x6fc2
	.uleb128 0x7
	.long	.LASF496
	.byte	0x18
	.byte	0xf
	.byte	0x4f
	.long	0x6dc8
	.uleb128 0x8
	.long	0x6c8d
	.byte	0
	.uleb128 0x9
	.long	.LASF497
	.byte	0xf
	.byte	0x52
	.long	0x6dc8
	.byte	0
	.uleb128 0x9
	.long	.LASF498
	.byte	0xf
	.byte	0x53
	.long	0x6dc8
	.byte	0x8
	.uleb128 0x9
	.long	.LASF499
	.byte	0xf
	.byte	0x54
	.long	0x6dc8
	.byte	0x10
	.uleb128 0xa
	.long	.LASF496
	.byte	0xf
	.byte	0x56
	.long	.LASF980
	.long	0x6d4f
	.long	0x6d55
	.uleb128 0xb
	.long	0x14427
	.byte	0
	.uleb128 0xa
	.long	.LASF496
	.byte	0xf
	.byte	0x5a
	.long	.LASF981
	.long	0x6d68
	.long	0x6d73
	.uleb128 0xb
	.long	0x14427
	.uleb128 0xc
	.long	0x1442d
	.byte	0
	.uleb128 0xa
	.long	.LASF496
	.byte	0xf
	.byte	0x5f
	.long	.LASF982
	.long	0x6d86
	.long	0x6d91
	.uleb128 0xb
	.long	0x14427
	.uleb128 0xc
	.long	0x14433
	.byte	0
	.uleb128 0xa
	.long	.LASF503
	.byte	0xf
	.byte	0x65
	.long	.LASF983
	.long	0x6da4
	.long	0x6daf
	.uleb128 0xb
	.long	0x14427
	.uleb128 0xc
	.long	0x14439
	.byte	0
	.uleb128 0xd
	.long	.LASF696
	.long	.LASF984
	.long	0x6dbc
	.uleb128 0xb
	.long	0x14427
	.uleb128 0xb
	.long	0x30
	.byte	0
	.byte	0
	.uleb128 0x16
	.long	.LASF4
	.byte	0xf
	.byte	0x4d
	.long	0x8e9b
	.uleb128 0x16
	.long	.LASF505
	.byte	0xf
	.byte	0x4b
	.long	0x8f5a
	.uleb128 0x14
	.long	0x6dd3
	.uleb128 0x9
	.long	.LASF506
	.byte	0xf
	.byte	0xa4
	.long	0x6d06
	.byte	0
	.uleb128 0x16
	.long	.LASF9
	.byte	0xf
	.byte	0x6e
	.long	0x6c8d
	.uleb128 0x17
	.long	.LASF507
	.byte	0xf
	.byte	0x71
	.long	.LASF985
	.long	0x1443f
	.long	0x6e11
	.long	0x6e17
	.uleb128 0xb
	.long	0x14445
	.byte	0
	.uleb128 0x17
	.long	.LASF507
	.byte	0xf
	.byte	0x75
	.long	.LASF986
	.long	0x1442d
	.long	0x6e2e
	.long	0x6e34
	.uleb128 0xb
	.long	0x1444b
	.byte	0
	.uleb128 0x17
	.long	.LASF211
	.byte	0xf
	.byte	0x79
	.long	.LASF987
	.long	0x6def
	.long	0x6e4b
	.long	0x6e51
	.uleb128 0xb
	.long	0x1444b
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x7c
	.long	.LASF988
	.long	0x6e64
	.long	0x6e6a
	.uleb128 0xb
	.long	0x14445
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x7f
	.long	.LASF989
	.long	0x6e7d
	.long	0x6e88
	.uleb128 0xb
	.long	0x14445
	.uleb128 0xc
	.long	0x14451
	.byte	0
	.uleb128 0x14
	.long	0x6def
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x82
	.long	.LASF990
	.long	0x6ea0
	.long	0x6eab
	.uleb128 0xb
	.long	0x14445
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x86
	.long	.LASF991
	.long	0x6ebe
	.long	0x6ece
	.uleb128 0xb
	.long	0x14445
	.uleb128 0xc
	.long	0x21c6
	.uleb128 0xc
	.long	0x14451
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x8b
	.long	.LASF992
	.long	0x6ee1
	.long	0x6eec
	.uleb128 0xb
	.long	0x14445
	.uleb128 0xc
	.long	0x14433
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x8e
	.long	.LASF993
	.long	0x6eff
	.long	0x6f0a
	.uleb128 0xb
	.long	0x14445
	.uleb128 0xc
	.long	0x14457
	.byte	0
	.uleb128 0xa
	.long	.LASF511
	.byte	0xf
	.byte	0x92
	.long	.LASF994
	.long	0x6f1d
	.long	0x6f2d
	.uleb128 0xb
	.long	0x14445
	.uleb128 0xc
	.long	0x14457
	.uleb128 0xc
	.long	0x14451
	.byte	0
	.uleb128 0xa
	.long	.LASF519
	.byte	0xf
	.byte	0x9f
	.long	.LASF995
	.long	0x6f40
	.long	0x6f4b
	.uleb128 0xb
	.long	0x14445
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x17
	.long	.LASF521
	.byte	0xf
	.byte	0xa7
	.long	.LASF996
	.long	0x6dc8
	.long	0x6f62
	.long	0x6f6d
	.uleb128 0xb
	.long	0x14445
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0xa
	.long	.LASF523
	.byte	0xf
	.byte	0xae
	.long	.LASF997
	.long	0x6f80
	.long	0x6f90
	.uleb128 0xb
	.long	0x14445
	.uleb128 0xc
	.long	0x6dc8
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0x26
	.long	.LASF525
	.byte	0xf
	.byte	0xb7
	.long	.LASF998
	.byte	0x3
	.long	0x6fa4
	.long	0x6faf
	.uleb128 0xb
	.long	0x14445
	.uleb128 0xc
	.long	0x21c6
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0xc3f6
	.uleb128 0x20
	.long	.LASF261
	.long	0x6c8d
	.byte	0
	.uleb128 0x14
	.long	0x6cfa
	.uleb128 0x2a
	.long	.LASF999
	.uleb128 0x2a
	.long	.LASF1000
	.uleb128 0x6
	.long	.LASF1001
	.byte	0x10
	.byte	0x24
	.byte	0x2f
	.long	0x70b9
	.uleb128 0xe
	.long	.LASF13
	.byte	0x24
	.byte	0x36
	.long	0x1406a
	.byte	0x1
	.uleb128 0x9
	.long	.LASF349
	.byte	0x24
	.byte	0x3a
	.long	0x6fdd
	.byte	0
	.uleb128 0xe
	.long	.LASF5
	.byte	0x24
	.byte	0x35
	.long	0x21c6
	.byte	0x1
	.uleb128 0x9
	.long	.LASF350
	.byte	0x24
	.byte	0x3b
	.long	0x6ff5
	.byte	0x8
	.uleb128 0xe
	.long	.LASF14
	.byte	0x24
	.byte	0x37
	.long	0x1406a
	.byte	0x1
	.uleb128 0xa
	.long	.LASF351
	.byte	0x24
	.byte	0x3e
	.long	.LASF1002
	.long	0x702c
	.long	0x703c
	.uleb128 0xb
	.long	0x1485a
	.uleb128 0xc
	.long	0x700d
	.uleb128 0xc
	.long	0x6ff5
	.byte	0
	.uleb128 0x26
	.long	.LASF351
	.byte	0x24
	.byte	0x42
	.long	.LASF1003
	.byte	0x1
	.long	0x7050
	.long	0x7056
	.uleb128 0xb
	.long	0x1485a
	.byte	0
	.uleb128 0x27
	.long	.LASF115
	.byte	0x24
	.byte	0x47
	.long	.LASF1004
	.long	0x6ff5
	.byte	0x1
	.long	0x706e
	.long	0x7074
	.uleb128 0xb
	.long	0x14860
	.byte	0
	.uleb128 0x27
	.long	.LASF96
	.byte	0x24
	.byte	0x4b
	.long	.LASF1005
	.long	0x700d
	.byte	0x1
	.long	0x708c
	.long	0x7092
	.uleb128 0xb
	.long	0x14860
	.byte	0
	.uleb128 0x38
	.string	"end"
	.byte	0x24
	.byte	0x4f
	.long	.LASF1006
	.long	0x700d
	.byte	0x1
	.long	0x70aa
	.long	0x70b0
	.uleb128 0xb
	.long	0x14860
	.byte	0
	.uleb128 0x2c
	.string	"_E"
	.long	0xc3f6
	.byte	0
	.uleb128 0x14
	.long	0x60c2
	.uleb128 0x16
	.long	.LASF1007
	.byte	0x34
	.byte	0x8d
	.long	0x2a30
	.uleb128 0x14
	.long	0x51b3
	.uleb128 0x14
	.long	0x6fd1
	.uleb128 0x7
	.long	.LASF1008
	.byte	0x1
	.byte	0x1d
	.byte	0xb2
	.long	0x710a
	.uleb128 0x16
	.long	.LASF619
	.byte	0x1d
	.byte	0xb6
	.long	0x22b2
	.uleb128 0x16
	.long	.LASF4
	.byte	0x1d
	.byte	0xb7
	.long	0x919c
	.uleb128 0x16
	.long	.LASF10
	.byte	0x1d
	.byte	0xb8
	.long	0xa1fc
	.uleb128 0x20
	.long	.LASF620
	.long	0x919c
	.byte	0
	.uleb128 0x7
	.long	.LASF1009
	.byte	0x1
	.byte	0x1d
	.byte	0xbd
	.long	0x714c
	.uleb128 0x16
	.long	.LASF1010
	.byte	0x1d
	.byte	0xbf
	.long	0x1fe4
	.uleb128 0x16
	.long	.LASF619
	.byte	0x1d
	.byte	0xc1
	.long	0x22b2
	.uleb128 0x16
	.long	.LASF4
	.byte	0x1d
	.byte	0xc2
	.long	0x9472
	.uleb128 0x16
	.long	.LASF10
	.byte	0x1d
	.byte	0xc3
	.long	0xa202
	.uleb128 0x20
	.long	.LASF620
	.long	0x9472
	.byte	0
	.uleb128 0x7
	.long	.LASF1011
	.byte	0x1
	.byte	0x35
	.byte	0x69
	.long	0x716d
	.uleb128 0x16
	.long	.LASF1012
	.byte	0x35
	.byte	0x6b
	.long	0x91a2
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x91a2
	.byte	0
	.uleb128 0x7
	.long	.LASF1013
	.byte	0x1
	.byte	0x35
	.byte	0x96
	.long	0x71a7
	.uleb128 0x16
	.long	.LASF4
	.byte	0x35
	.byte	0x99
	.long	0x919c
	.uleb128 0x57
	.long	.LASF1014
	.byte	0x35
	.byte	0xa8
	.long	.LASF1015
	.long	0x7179
	.long	0x719d
	.uleb128 0xc
	.long	0x1491b
	.byte	0
	.uleb128 0x20
	.long	.LASF1016
	.long	0x919c
	.byte	0
	.uleb128 0x7
	.long	.LASF1017
	.byte	0x1
	.byte	0x35
	.byte	0x69
	.long	0x71c8
	.uleb128 0x16
	.long	.LASF1012
	.byte	0x35
	.byte	0x6b
	.long	0x9478
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x9478
	.byte	0
	.uleb128 0x7
	.long	.LASF1018
	.byte	0x1
	.byte	0x35
	.byte	0x96
	.long	0x7202
	.uleb128 0x16
	.long	.LASF4
	.byte	0x35
	.byte	0x99
	.long	0x9472
	.uleb128 0x57
	.long	.LASF1014
	.byte	0x35
	.byte	0xa8
	.long	.LASF1019
	.long	0x71d4
	.long	0x71f8
	.uleb128 0xc
	.long	0x14951
	.byte	0
	.uleb128 0x20
	.long	.LASF1016
	.long	0x9472
	.byte	0
	.uleb128 0x7
	.long	.LASF1020
	.byte	0x1
	.byte	0x10
	.byte	0x60
	.long	0x722e
	.uleb128 0x48
	.long	.LASF1021
	.byte	0x10
	.byte	0x64
	.long	.LASF1023
	.uleb128 0x20
	.long	.LASF446
	.long	0x14013
	.uleb128 0xc
	.long	0x14013
	.uleb128 0xc
	.long	0x14013
	.byte	0
	.byte	0
	.uleb128 0x7
	.long	.LASF1024
	.byte	0x1
	.byte	0x1d
	.byte	0xbd
	.long	0x7265
	.uleb128 0x16
	.long	.LASF619
	.byte	0x1d
	.byte	0xc1
	.long	0x22b2
	.uleb128 0x16
	.long	.LASF4
	.byte	0x1d
	.byte	0xc2
	.long	0x9720
	.uleb128 0x16
	.long	.LASF10
	.byte	0x1d
	.byte	0xc3
	.long	0x13fbc
	.uleb128 0x20
	.long	.LASF620
	.long	0x9720
	.byte	0
	.uleb128 0x36
	.long	.LASF1025
	.byte	0x1
	.byte	0x12
	.value	0x1fd
	.long	0x72ae
	.uleb128 0x1b
	.long	.LASF1026
	.byte	0x12
	.value	0x201
	.long	.LASF1027
	.long	0x14013
	.long	0x72a3
	.uleb128 0x20
	.long	.LASF446
	.long	0x14013
	.uleb128 0x20
	.long	.LASF452
	.long	0x9126
	.uleb128 0xc
	.long	0x14013
	.uleb128 0xc
	.long	0x9126
	.byte	0
	.uleb128 0x58
	.long	.LASF1028
	.long	0x9ef1
	.byte	0
	.byte	0
	.uleb128 0x7
	.long	.LASF1029
	.byte	0x1
	.byte	0x1d
	.byte	0xd4
	.long	0x72f2
	.uleb128 0x16
	.long	.LASF1030
	.byte	0x1d
	.byte	0xd6
	.long	0x14128
	.uleb128 0x57
	.long	.LASF1031
	.byte	0x1d
	.byte	0xd7
	.long	.LASF1032
	.long	0x72ba
	.long	0x72de
	.uleb128 0xc
	.long	0x14128
	.byte	0
	.uleb128 0x20
	.long	.LASF620
	.long	0x14128
	.uleb128 0x58
	.long	.LASF1033
	.long	0x9ef1
	.byte	0
	.byte	0
	.uleb128 0x3e
	.long	.LASF1034
	.long	0x734b
	.uleb128 0x27
	.long	.LASF1035
	.byte	0x36
	.byte	0x89
	.long	.LASF1036
	.long	0x27fd
	.byte	0x1
	.long	0x7313
	.long	0x7319
	.uleb128 0xb
	.long	0x14a5a
	.byte	0
	.uleb128 0x26
	.long	.LASF1037
	.byte	0x36
	.byte	0x9d
	.long	.LASF1038
	.byte	0x1
	.long	0x732d
	.long	0x7338
	.uleb128 0xb
	.long	0x14b5c
	.uleb128 0xc
	.long	0x27fd
	.byte	0
	.uleb128 0x20
	.long	.LASF259
	.long	0x91a2
	.uleb128 0x21
	.long	.LASF260
	.long	0x1ffe
	.byte	0
	.uleb128 0x14
	.long	0x72f2
	.uleb128 0x57
	.long	.LASF1039
	.byte	0x26
	.byte	0xa9
	.long	.LASF1040
	.long	0x2725
	.long	0x736e
	.uleb128 0xc
	.long	0x2725
	.uleb128 0xc
	.long	0x2725
	.byte	0
	.uleb128 0x1b
	.long	.LASF1041
	.byte	0xc
	.value	0x22c
	.long	.LASF1042
	.long	0xa6cb
	.long	0x7396
	.uleb128 0x20
	.long	.LASF260
	.long	0x1ffe
	.uleb128 0xc
	.long	0xa6cb
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1b
	.long	.LASF1043
	.byte	0xe
	.value	0x2ee
	.long	.LASF1044
	.long	0xb43b
	.long	0x73b0
	.uleb128 0xc
	.long	0xb43b
	.byte	0
	.uleb128 0x1b
	.long	.LASF1045
	.byte	0xe
	.value	0x176
	.long	.LASF1046
	.long	0x3f1e
	.long	0x73d8
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.uleb128 0xc
	.long	0xb334
	.uleb128 0xc
	.long	0xb423
	.byte	0
	.uleb128 0x57
	.long	.LASF1047
	.byte	0x6
	.byte	0xc3
	.long	.LASF1048
	.long	0x13fbc
	.long	0x73ff
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x30
	.uleb128 0xc
	.long	0x13fbc
	.uleb128 0xc
	.long	0x13fbc
	.byte	0
	.uleb128 0x57
	.long	.LASF1049
	.byte	0x6
	.byte	0xdb
	.long	.LASF1050
	.long	0xb334
	.long	0x7426
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.uleb128 0xc
	.long	0xb334
	.uleb128 0xc
	.long	0xb334
	.byte	0
	.uleb128 0x31
	.string	"abs"
	.byte	0x2b
	.byte	0x51
	.long	.LASF1051
	.long	0x29
	.long	0x743f
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1b
	.long	.LASF1052
	.byte	0xe
	.value	0x1b2
	.long	.LASF1053
	.long	0x3f1e
	.long	0x7467
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.uleb128 0xc
	.long	0xb334
	.uleb128 0xc
	.long	0xb423
	.byte	0
	.uleb128 0x1b
	.long	.LASF1054
	.byte	0xe
	.value	0x2f6
	.long	.LASF1055
	.long	0x3f1e
	.long	0x748a
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.uleb128 0xc
	.long	0xb423
	.byte	0
	.uleb128 0x1b
	.long	.LASF1056
	.byte	0xe
	.value	0x182
	.long	.LASF1057
	.long	0x3f1e
	.long	0x74b2
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.uleb128 0xc
	.long	0xb423
	.uleb128 0xc
	.long	0xb423
	.byte	0
	.uleb128 0x1b
	.long	.LASF1058
	.byte	0xe
	.value	0x21c
	.long	.LASF1059
	.long	0x29
	.long	0x74d5
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.uleb128 0xc
	.long	0xb423
	.byte	0
	.uleb128 0x30
	.long	.LASF1060
	.byte	0x10
	.byte	0x7a
	.long	.LASF1061
	.long	0x74f8
	.uleb128 0x20
	.long	.LASF446
	.long	0xaa8a
	.uleb128 0xc
	.long	0xaa8a
	.uleb128 0xc
	.long	0xaa8a
	.byte	0
	.uleb128 0x30
	.long	.LASF1062
	.byte	0x10
	.byte	0x94
	.long	.LASF1063
	.long	0x7529
	.uleb128 0x20
	.long	.LASF446
	.long	0xaa8a
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x30
	.uleb128 0xc
	.long	0xaa8a
	.uleb128 0xc
	.long	0xaa8a
	.uleb128 0xc
	.long	0x1420b
	.byte	0
	.uleb128 0x30
	.long	.LASF1064
	.byte	0x10
	.byte	0x7a
	.long	.LASF1065
	.long	0x754c
	.uleb128 0x20
	.long	.LASF446
	.long	0x14128
	.uleb128 0xc
	.long	0x14128
	.uleb128 0xc
	.long	0x14128
	.byte	0
	.uleb128 0x30
	.long	.LASF1066
	.byte	0x10
	.byte	0x94
	.long	.LASF1067
	.long	0x757d
	.uleb128 0x20
	.long	.LASF446
	.long	0x14128
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x912d
	.uleb128 0xc
	.long	0x14128
	.uleb128 0xc
	.long	0x14128
	.uleb128 0xc
	.long	0x14157
	.byte	0
	.uleb128 0x57
	.long	.LASF1068
	.byte	0x37
	.byte	0x2f
	.long	.LASF1069
	.long	0x919c
	.long	0x759f
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x91a2
	.uleb128 0xc
	.long	0xa1fc
	.byte	0
	.uleb128 0x57
	.long	.LASF1070
	.byte	0x37
	.byte	0x87
	.long	.LASF1071
	.long	0x919c
	.long	0x75c1
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x91a2
	.uleb128 0xc
	.long	0xa1fc
	.byte	0
	.uleb128 0x30
	.long	.LASF1072
	.byte	0x10
	.byte	0x5c
	.long	.LASF1073
	.long	0x75df
	.uleb128 0x2c
	.string	"_Tp"
	.long	0xc3f6
	.uleb128 0xc
	.long	0x14013
	.byte	0
	.uleb128 0x57
	.long	.LASF1074
	.byte	0x37
	.byte	0x2f
	.long	.LASF1075
	.long	0x14013
	.long	0x7601
	.uleb128 0x2c
	.string	"_Tp"
	.long	0xc3f6
	.uleb128 0xc
	.long	0x1405e
	.byte	0
	.uleb128 0x30
	.long	.LASF1076
	.byte	0x10
	.byte	0x7a
	.long	.LASF1077
	.long	0x7624
	.uleb128 0x20
	.long	.LASF446
	.long	0x14013
	.uleb128 0xc
	.long	0x14013
	.uleb128 0xc
	.long	0x14013
	.byte	0
	.uleb128 0x30
	.long	.LASF1078
	.byte	0x10
	.byte	0x94
	.long	.LASF1079
	.long	0x7655
	.uleb128 0x20
	.long	.LASF446
	.long	0x14013
	.uleb128 0x2c
	.string	"_Tp"
	.long	0xc3f6
	.uleb128 0xc
	.long	0x14013
	.uleb128 0xc
	.long	0x14013
	.uleb128 0xc
	.long	0x14409
	.byte	0
	.uleb128 0x57
	.long	.LASF1080
	.byte	0x37
	.byte	0x2f
	.long	.LASF1081
	.long	0x9472
	.long	0x7677
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x9478
	.uleb128 0xc
	.long	0xa202
	.byte	0
	.uleb128 0x57
	.long	.LASF1082
	.byte	0x37
	.byte	0x87
	.long	.LASF1083
	.long	0x9472
	.long	0x7699
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x9478
	.uleb128 0xc
	.long	0xa202
	.byte	0
	.uleb128 0x57
	.long	.LASF1084
	.byte	0x1d
	.byte	0xcc
	.long	.LASF1085
	.long	0x7116
	.long	0x76bb
	.uleb128 0x20
	.long	.LASF1086
	.long	0x9472
	.uleb128 0xc
	.long	0x1486c
	.byte	0
	.uleb128 0x57
	.long	.LASF1087
	.byte	0x38
	.byte	0x5a
	.long	.LASF1088
	.long	0x7121
	.long	0x76e7
	.uleb128 0x20
	.long	.LASF1089
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x1fe4
	.byte	0
	.uleb128 0x57
	.long	.LASF1090
	.byte	0x38
	.byte	0x72
	.long	.LASF1091
	.long	0x7121
	.long	0x770e
	.uleb128 0x20
	.long	.LASF1092
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x30
	.long	.LASF1093
	.byte	0x10
	.byte	0x4a
	.long	.LASF1094
	.long	0x7731
	.uleb128 0x2c
	.string	"_T1"
	.long	0xc3f6
	.uleb128 0x59
	.long	.LASF2874
	.uleb128 0xc
	.long	0x14013
	.byte	0
	.uleb128 0x1b
	.long	.LASF1095
	.byte	0x12
	.value	0x236
	.long	.LASF1096
	.long	0x14013
	.long	0x7762
	.uleb128 0x20
	.long	.LASF446
	.long	0x14013
	.uleb128 0x20
	.long	.LASF452
	.long	0x9126
	.uleb128 0xc
	.long	0x14013
	.uleb128 0xc
	.long	0x9126
	.byte	0
	.uleb128 0x1b
	.long	.LASF1097
	.byte	0x12
	.value	0x27b
	.long	.LASF1098
	.long	0x14013
	.long	0x77a1
	.uleb128 0x20
	.long	.LASF446
	.long	0x14013
	.uleb128 0x20
	.long	.LASF452
	.long	0x9126
	.uleb128 0x2c
	.string	"_Tp"
	.long	0xc3f6
	.uleb128 0xc
	.long	0x14013
	.uleb128 0xc
	.long	0x9126
	.uleb128 0xc
	.long	0x14409
	.byte	0
	.uleb128 0x1a
	.long	.LASF1099
	.byte	0x6
	.value	0x2cf
	.long	.LASF1100
	.long	0x77ca
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x912d
	.uleb128 0xc
	.long	0x14128
	.uleb128 0xc
	.long	0x14128
	.uleb128 0xc
	.long	0x14163
	.byte	0
	.uleb128 0x1b
	.long	.LASF1101
	.byte	0x6
	.value	0x309
	.long	.LASF1102
	.long	0x90f0
	.long	0x7800
	.uleb128 0x20
	.long	.LASF452
	.long	0x9126
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x912d
	.uleb128 0xc
	.long	0x14128
	.uleb128 0xc
	.long	0x9126
	.uleb128 0xc
	.long	0x14163
	.byte	0
	.uleb128 0x1b
	.long	.LASF1103
	.byte	0x6
	.value	0x11a
	.long	.LASF1104
	.long	0x72ba
	.long	0x7823
	.uleb128 0x20
	.long	.LASF620
	.long	0x14128
	.uleb128 0xc
	.long	0x14128
	.byte	0
	.uleb128 0x1b
	.long	.LASF1105
	.byte	0x6
	.value	0x320
	.long	.LASF1106
	.long	0x14128
	.long	0x7862
	.uleb128 0x2c
	.string	"_OI"
	.long	0x14128
	.uleb128 0x20
	.long	.LASF452
	.long	0x9126
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x912d
	.uleb128 0xc
	.long	0x14128
	.uleb128 0xc
	.long	0x9126
	.uleb128 0xc
	.long	0x14163
	.byte	0
	.uleb128 0x1b
	.long	.LASF1107
	.byte	0x12
	.value	0x236
	.long	.LASF1108
	.long	0x14128
	.long	0x7893
	.uleb128 0x20
	.long	.LASF446
	.long	0x14128
	.uleb128 0x20
	.long	.LASF452
	.long	0x9126
	.uleb128 0xc
	.long	0x14128
	.uleb128 0xc
	.long	0x9126
	.byte	0
	.uleb128 0x1b
	.long	.LASF1109
	.byte	0x12
	.value	0x27b
	.long	.LASF1110
	.long	0x14128
	.long	0x78d2
	.uleb128 0x20
	.long	.LASF446
	.long	0x14128
	.uleb128 0x20
	.long	.LASF452
	.long	0x9126
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x912d
	.uleb128 0xc
	.long	0x14128
	.uleb128 0xc
	.long	0x9126
	.uleb128 0xc
	.long	0x14157
	.byte	0
	.uleb128 0x5a
	.long	.LASF1115
	.byte	0x1c
	.byte	0x4f
	.long	0x78df
	.byte	0x1
	.byte	0
	.uleb128 0x14
	.long	0x1fae
	.uleb128 0x5b
	.long	.LASF1111
	.byte	0x20
	.byte	0x54
	.long	.LASF1113
	.long	0x78f3
	.uleb128 0x14
	.long	0x22aa
	.uleb128 0x5b
	.long	.LASF1112
	.byte	0x14
	.byte	0x3d
	.long	.LASF1114
	.long	0x70be
	.uleb128 0x5c
	.long	.LASF2885
	.byte	0x14
	.byte	0x4a
	.long	0x279f
	.uleb128 0x5a
	.long	.LASF1116
	.byte	0x32
	.byte	0x29
	.long	0x791f
	.byte	0x1
	.byte	0
	.uleb128 0x14
	.long	0x437b
	.uleb128 0x5d
	.long	.LASF1117
	.byte	0x33
	.value	0x47a
	.long	0x7932
	.byte	0x1
	.byte	0
	.uleb128 0x14
	.long	0x4383
	.uleb128 0x46
	.long	.LASF1118
	.long	.LASF1119
	.byte	0x39
	.byte	0x3f
	.long	.LASF1118
	.uleb128 0x46
	.long	.LASF1120
	.long	.LASF1121
	.byte	0x3a
	.byte	0x4c
	.long	.LASF1120
	.uleb128 0x46
	.long	.LASF1122
	.long	.LASF1123
	.byte	0x39
	.byte	0x34
	.long	.LASF1122
	.byte	0
	.uleb128 0x5
	.long	.LASF1124
	.byte	0x18
	.byte	0xdd
	.long	0x911b
	.uleb128 0x2f
	.long	.LASF1
	.byte	0x18
	.byte	0xde
	.uleb128 0x22
	.byte	0x18
	.byte	0xde
	.long	0x7970
	.uleb128 0x23
	.byte	0x17
	.byte	0xf8
	.long	0x9e5e
	.uleb128 0x24
	.byte	0x17
	.value	0x101
	.long	0x9e80
	.uleb128 0x24
	.byte	0x17
	.value	0x102
	.long	0x9ea7
	.uleb128 0x2f
	.long	.LASF1125
	.byte	0x3b
	.byte	0x24
	.uleb128 0x23
	.byte	0x11
	.byte	0x2c
	.long	0x21c6
	.uleb128 0x23
	.byte	0x11
	.byte	0x2d
	.long	0x22b2
	.uleb128 0x6
	.long	.LASF1126
	.byte	0x1
	.byte	0x11
	.byte	0x3a
	.long	0x7b04
	.uleb128 0xe
	.long	.LASF5
	.byte	0x11
	.byte	0x3d
	.long	0x21c6
	.byte	0x1
	.uleb128 0xe
	.long	.LASF4
	.byte	0x11
	.byte	0x3f
	.long	0x919c
	.byte	0x1
	.uleb128 0xe
	.long	.LASF12
	.byte	0x11
	.byte	0x40
	.long	0x9472
	.byte	0x1
	.uleb128 0xe
	.long	.LASF10
	.byte	0x11
	.byte	0x41
	.long	0xa1fc
	.byte	0x1
	.uleb128 0xe
	.long	.LASF11
	.byte	0x11
	.byte	0x42
	.long	0xa202
	.byte	0x1
	.uleb128 0x26
	.long	.LASF1127
	.byte	0x11
	.byte	0x4f
	.long	.LASF1128
	.byte	0x1
	.long	0x7a06
	.long	0x7a0c
	.uleb128 0xb
	.long	0xa208
	.byte	0
	.uleb128 0x26
	.long	.LASF1127
	.byte	0x11
	.byte	0x51
	.long	.LASF1129
	.byte	0x1
	.long	0x7a20
	.long	0x7a2b
	.uleb128 0xb
	.long	0xa208
	.uleb128 0xc
	.long	0xa20e
	.byte	0
	.uleb128 0x26
	.long	.LASF1130
	.byte	0x11
	.byte	0x56
	.long	.LASF1131
	.byte	0x1
	.long	0x7a3f
	.long	0x7a4a
	.uleb128 0xb
	.long	0xa208
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x27
	.long	.LASF1132
	.byte	0x11
	.byte	0x59
	.long	.LASF1133
	.long	0x79c2
	.byte	0x1
	.long	0x7a62
	.long	0x7a6d
	.uleb128 0xb
	.long	0xa214
	.uleb128 0xc
	.long	0x79da
	.byte	0
	.uleb128 0x27
	.long	.LASF1132
	.byte	0x11
	.byte	0x5d
	.long	.LASF1134
	.long	0x79ce
	.byte	0x1
	.long	0x7a85
	.long	0x7a90
	.uleb128 0xb
	.long	0xa214
	.uleb128 0xc
	.long	0x79e6
	.byte	0
	.uleb128 0x27
	.long	.LASF335
	.byte	0x11
	.byte	0x63
	.long	.LASF1135
	.long	0x79c2
	.byte	0x1
	.long	0x7aa8
	.long	0x7ab8
	.uleb128 0xb
	.long	0xa208
	.uleb128 0xc
	.long	0x79b6
	.uleb128 0xc
	.long	0xa1f5
	.byte	0
	.uleb128 0x26
	.long	.LASF338
	.byte	0x11
	.byte	0x6d
	.long	.LASF1136
	.byte	0x1
	.long	0x7acc
	.long	0x7adc
	.uleb128 0xb
	.long	0xa208
	.uleb128 0xc
	.long	0x79c2
	.uleb128 0xc
	.long	0x79b6
	.byte	0
	.uleb128 0x27
	.long	.LASF119
	.byte	0x11
	.byte	0x71
	.long	.LASF1137
	.long	0x79b6
	.byte	0x1
	.long	0x7af4
	.long	0x7afa
	.uleb128 0xb
	.long	0xa214
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x91a2
	.byte	0
	.uleb128 0x14
	.long	0x79aa
	.uleb128 0x7
	.long	.LASF1138
	.byte	0x1
	.byte	0x3c
	.byte	0x37
	.long	0x7b4b
	.uleb128 0x2d
	.long	.LASF1139
	.byte	0x3c
	.byte	0x3a
	.long	0x9726
	.uleb128 0x2d
	.long	.LASF1140
	.byte	0x3c
	.byte	0x3b
	.long	0x9726
	.uleb128 0x2d
	.long	.LASF1141
	.byte	0x3c
	.byte	0x3f
	.long	0x9efe
	.uleb128 0x2d
	.long	.LASF1142
	.byte	0x3c
	.byte	0x40
	.long	0x9726
	.uleb128 0x20
	.long	.LASF1143
	.long	0x30
	.byte	0
	.uleb128 0x23
	.byte	0x22
	.byte	0xd6
	.long	0xa2ab
	.uleb128 0x23
	.byte	0x22
	.byte	0xe6
	.long	0xa528
	.uleb128 0x23
	.byte	0x22
	.byte	0xf1
	.long	0xa543
	.uleb128 0x23
	.byte	0x22
	.byte	0xf2
	.long	0xa559
	.uleb128 0x23
	.byte	0x22
	.byte	0xf3
	.long	0xa578
	.uleb128 0x23
	.byte	0x22
	.byte	0xf5
	.long	0xa597
	.uleb128 0x23
	.byte	0x22
	.byte	0xf6
	.long	0xa5b1
	.uleb128 0x31
	.string	"div"
	.byte	0x22
	.byte	0xe3
	.long	.LASF1144
	.long	0xa2ab
	.long	0x7b9a
	.uleb128 0xc
	.long	0x9ea0
	.uleb128 0xc
	.long	0x9ea0
	.byte	0
	.uleb128 0x7
	.long	.LASF1145
	.byte	0x1
	.byte	0x3d
	.byte	0x5f
	.long	0x7cc0
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x2490
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x24b4
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x24d4
	.uleb128 0x8
	.long	0x241c
	.byte	0
	.uleb128 0x16
	.long	.LASF290
	.byte	0x3d
	.byte	0x67
	.long	0x2435
	.uleb128 0x16
	.long	.LASF4
	.byte	0x3d
	.byte	0x68
	.long	0x2441
	.uleb128 0x16
	.long	.LASF12
	.byte	0x3d
	.byte	0x69
	.long	0x244d
	.uleb128 0x16
	.long	.LASF5
	.byte	0x3d
	.byte	0x6a
	.long	0x2465
	.uleb128 0x16
	.long	.LASF10
	.byte	0x3d
	.byte	0x6d
	.long	0xa5d7
	.uleb128 0x16
	.long	.LASF11
	.byte	0x3d
	.byte	0x6e
	.long	0xa5dd
	.uleb128 0x14
	.long	0x7bc1
	.uleb128 0x57
	.long	.LASF1146
	.byte	0x3d
	.byte	0x8b
	.long	.LASF1147
	.long	0x22c8
	.long	0x7c21
	.uleb128 0xc
	.long	0xa220
	.byte	0
	.uleb128 0x30
	.long	.LASF1148
	.byte	0x3d
	.byte	0x8e
	.long	.LASF1149
	.long	0x7c3b
	.uleb128 0xc
	.long	0xa5e3
	.uleb128 0xc
	.long	0xa5e3
	.byte	0
	.uleb128 0x5e
	.long	.LASF1150
	.byte	0x3d
	.byte	0x91
	.long	.LASF1152
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1151
	.byte	0x3d
	.byte	0x94
	.long	.LASF1153
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1154
	.byte	0x3d
	.byte	0x97
	.long	.LASF1155
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1156
	.byte	0x3d
	.byte	0x9a
	.long	.LASF1157
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1158
	.byte	0x3d
	.byte	0x9d
	.long	.LASF1159
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1160
	.byte	0x3d
	.byte	0xa0
	.long	.LASF1161
	.long	0x9ef1
	.uleb128 0x7
	.long	.LASF1162
	.byte	0x1
	.byte	0x3d
	.byte	0xa8
	.long	0x7cb6
	.uleb128 0x16
	.long	.LASF1163
	.byte	0x3d
	.byte	0xa9
	.long	0x250d
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x91a2
	.byte	0
	.uleb128 0x20
	.long	.LASF261
	.long	0x22c8
	.byte	0
	.uleb128 0x54
	.long	.LASF1164
	.byte	0x8
	.byte	0x3e
	.value	0x2d1
	.long	0x7ee7
	.uleb128 0x51
	.long	.LASF1165
	.byte	0x3e
	.value	0x2d4
	.long	0x919c
	.byte	0
	.byte	0x2
	.uleb128 0x42
	.long	.LASF619
	.byte	0x3e
	.value	0x2dc
	.long	0x70df
	.byte	0x1
	.uleb128 0x42
	.long	.LASF10
	.byte	0x3e
	.value	0x2dd
	.long	0x70f5
	.byte	0x1
	.uleb128 0x42
	.long	.LASF4
	.byte	0x3e
	.value	0x2de
	.long	0x70ea
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF1166
	.byte	0x3e
	.value	0x2e0
	.long	.LASF1167
	.byte	0x1
	.long	0x7d17
	.long	0x7d1d
	.uleb128 0xb
	.long	0x1487e
	.byte	0
	.uleb128 0x1d
	.long	.LASF1166
	.byte	0x3e
	.value	0x2e4
	.long	.LASF1168
	.byte	0x1
	.long	0x7d32
	.long	0x7d3d
	.uleb128 0xb
	.long	0x1487e
	.uleb128 0xc
	.long	0x14884
	.byte	0
	.uleb128 0x1e
	.long	.LASF1169
	.byte	0x3e
	.value	0x2f1
	.long	.LASF1170
	.long	0x7ce8
	.byte	0x1
	.long	0x7d56
	.long	0x7d5c
	.uleb128 0xb
	.long	0x1488f
	.byte	0
	.uleb128 0x1e
	.long	.LASF1171
	.byte	0x3e
	.value	0x2f5
	.long	.LASF1172
	.long	0x7cf5
	.byte	0x1
	.long	0x7d75
	.long	0x7d7b
	.uleb128 0xb
	.long	0x1488f
	.byte	0
	.uleb128 0x1e
	.long	.LASF1173
	.byte	0x3e
	.value	0x2f9
	.long	.LASF1174
	.long	0x14895
	.byte	0x1
	.long	0x7d94
	.long	0x7d9a
	.uleb128 0xb
	.long	0x1487e
	.byte	0
	.uleb128 0x1e
	.long	.LASF1173
	.byte	0x3e
	.value	0x300
	.long	.LASF1175
	.long	0x7cc0
	.byte	0x1
	.long	0x7db3
	.long	0x7dbe
	.uleb128 0xb
	.long	0x1487e
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1176
	.byte	0x3e
	.value	0x305
	.long	.LASF1177
	.long	0x14895
	.byte	0x1
	.long	0x7dd7
	.long	0x7ddd
	.uleb128 0xb
	.long	0x1487e
	.byte	0
	.uleb128 0x1e
	.long	.LASF1176
	.byte	0x3e
	.value	0x30c
	.long	.LASF1178
	.long	0x7cc0
	.byte	0x1
	.long	0x7df6
	.long	0x7e01
	.uleb128 0xb
	.long	0x1487e
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x3e
	.value	0x311
	.long	.LASF1179
	.long	0x7ce8
	.byte	0x1
	.long	0x7e1a
	.long	0x7e25
	.uleb128 0xb
	.long	0x1488f
	.uleb128 0xc
	.long	0x7cdb
	.byte	0
	.uleb128 0x1e
	.long	.LASF145
	.byte	0x3e
	.value	0x315
	.long	.LASF1180
	.long	0x14895
	.byte	0x1
	.long	0x7e3e
	.long	0x7e49
	.uleb128 0xb
	.long	0x1487e
	.uleb128 0xc
	.long	0x7cdb
	.byte	0
	.uleb128 0x1e
	.long	.LASF1181
	.byte	0x3e
	.value	0x319
	.long	.LASF1182
	.long	0x7cc0
	.byte	0x1
	.long	0x7e62
	.long	0x7e6d
	.uleb128 0xb
	.long	0x1488f
	.uleb128 0xc
	.long	0x7cdb
	.byte	0
	.uleb128 0x1e
	.long	.LASF637
	.byte	0x3e
	.value	0x31d
	.long	.LASF1183
	.long	0x14895
	.byte	0x1
	.long	0x7e86
	.long	0x7e91
	.uleb128 0xb
	.long	0x1487e
	.uleb128 0xc
	.long	0x7cdb
	.byte	0
	.uleb128 0x1e
	.long	.LASF1184
	.byte	0x3e
	.value	0x321
	.long	.LASF1185
	.long	0x7cc0
	.byte	0x1
	.long	0x7eaa
	.long	0x7eb5
	.uleb128 0xb
	.long	0x1488f
	.uleb128 0xc
	.long	0x7cdb
	.byte	0
	.uleb128 0x1e
	.long	.LASF1186
	.byte	0x3e
	.value	0x325
	.long	.LASF1187
	.long	0x14884
	.byte	0x1
	.long	0x7ece
	.long	0x7ed4
	.uleb128 0xb
	.long	0x1488f
	.byte	0
	.uleb128 0x20
	.long	.LASF620
	.long	0x919c
	.uleb128 0x20
	.long	.LASF1188
	.long	0x4d
	.byte	0
	.uleb128 0x54
	.long	.LASF1189
	.byte	0x8
	.byte	0x3e
	.value	0x2d1
	.long	0x810e
	.uleb128 0x51
	.long	.LASF1165
	.byte	0x3e
	.value	0x2d4
	.long	0x9472
	.byte	0
	.byte	0x2
	.uleb128 0x42
	.long	.LASF619
	.byte	0x3e
	.value	0x2dc
	.long	0x7121
	.byte	0x1
	.uleb128 0x42
	.long	.LASF10
	.byte	0x3e
	.value	0x2dd
	.long	0x7137
	.byte	0x1
	.uleb128 0x42
	.long	.LASF4
	.byte	0x3e
	.value	0x2de
	.long	0x712c
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF1166
	.byte	0x3e
	.value	0x2e0
	.long	.LASF1190
	.byte	0x1
	.long	0x7f3e
	.long	0x7f44
	.uleb128 0xb
	.long	0x14866
	.byte	0
	.uleb128 0x1d
	.long	.LASF1166
	.byte	0x3e
	.value	0x2e4
	.long	.LASF1191
	.byte	0x1
	.long	0x7f59
	.long	0x7f64
	.uleb128 0xb
	.long	0x14866
	.uleb128 0xc
	.long	0x1486c
	.byte	0
	.uleb128 0x1e
	.long	.LASF1169
	.byte	0x3e
	.value	0x2f1
	.long	.LASF1192
	.long	0x7f0f
	.byte	0x1
	.long	0x7f7d
	.long	0x7f83
	.uleb128 0xb
	.long	0x14872
	.byte	0
	.uleb128 0x1e
	.long	.LASF1171
	.byte	0x3e
	.value	0x2f5
	.long	.LASF1193
	.long	0x7f1c
	.byte	0x1
	.long	0x7f9c
	.long	0x7fa2
	.uleb128 0xb
	.long	0x14872
	.byte	0
	.uleb128 0x1e
	.long	.LASF1173
	.byte	0x3e
	.value	0x2f9
	.long	.LASF1194
	.long	0x14878
	.byte	0x1
	.long	0x7fbb
	.long	0x7fc1
	.uleb128 0xb
	.long	0x14866
	.byte	0
	.uleb128 0x1e
	.long	.LASF1173
	.byte	0x3e
	.value	0x300
	.long	.LASF1195
	.long	0x7ee7
	.byte	0x1
	.long	0x7fda
	.long	0x7fe5
	.uleb128 0xb
	.long	0x14866
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1176
	.byte	0x3e
	.value	0x305
	.long	.LASF1196
	.long	0x14878
	.byte	0x1
	.long	0x7ffe
	.long	0x8004
	.uleb128 0xb
	.long	0x14866
	.byte	0
	.uleb128 0x1e
	.long	.LASF1176
	.byte	0x3e
	.value	0x30c
	.long	.LASF1197
	.long	0x7ee7
	.byte	0x1
	.long	0x801d
	.long	0x8028
	.uleb128 0xb
	.long	0x14866
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x3e
	.value	0x311
	.long	.LASF1198
	.long	0x7f0f
	.byte	0x1
	.long	0x8041
	.long	0x804c
	.uleb128 0xb
	.long	0x14872
	.uleb128 0xc
	.long	0x7f02
	.byte	0
	.uleb128 0x1e
	.long	.LASF145
	.byte	0x3e
	.value	0x315
	.long	.LASF1199
	.long	0x14878
	.byte	0x1
	.long	0x8065
	.long	0x8070
	.uleb128 0xb
	.long	0x14866
	.uleb128 0xc
	.long	0x7f02
	.byte	0
	.uleb128 0x1e
	.long	.LASF1181
	.byte	0x3e
	.value	0x319
	.long	.LASF1200
	.long	0x7ee7
	.byte	0x1
	.long	0x8089
	.long	0x8094
	.uleb128 0xb
	.long	0x14872
	.uleb128 0xc
	.long	0x7f02
	.byte	0
	.uleb128 0x1e
	.long	.LASF637
	.byte	0x3e
	.value	0x31d
	.long	.LASF1201
	.long	0x14878
	.byte	0x1
	.long	0x80ad
	.long	0x80b8
	.uleb128 0xb
	.long	0x14866
	.uleb128 0xc
	.long	0x7f02
	.byte	0
	.uleb128 0x1e
	.long	.LASF1184
	.byte	0x3e
	.value	0x321
	.long	.LASF1202
	.long	0x7ee7
	.byte	0x1
	.long	0x80d1
	.long	0x80dc
	.uleb128 0xb
	.long	0x14872
	.uleb128 0xc
	.long	0x7f02
	.byte	0
	.uleb128 0x1e
	.long	.LASF1186
	.byte	0x3e
	.value	0x325
	.long	.LASF1203
	.long	0x1486c
	.byte	0x1
	.long	0x80f5
	.long	0x80fb
	.uleb128 0xb
	.long	0x14872
	.byte	0
	.uleb128 0x20
	.long	.LASF620
	.long	0x9472
	.uleb128 0x20
	.long	.LASF1188
	.long	0x4d
	.byte	0
	.uleb128 0x7
	.long	.LASF1204
	.byte	0x1
	.byte	0x3c
	.byte	0x64
	.long	0x8150
	.uleb128 0x2d
	.long	.LASF1205
	.byte	0x3c
	.byte	0x67
	.long	0x9726
	.uleb128 0x2d
	.long	.LASF1141
	.byte	0x3c
	.byte	0x6a
	.long	0x9efe
	.uleb128 0x2d
	.long	.LASF1206
	.byte	0x3c
	.byte	0x6b
	.long	0x9726
	.uleb128 0x2d
	.long	.LASF1207
	.byte	0x3c
	.byte	0x6c
	.long	0x9726
	.uleb128 0x20
	.long	.LASF1143
	.long	0x9c79
	.byte	0
	.uleb128 0x7
	.long	.LASF1208
	.byte	0x1
	.byte	0x3c
	.byte	0x64
	.long	0x8192
	.uleb128 0x2d
	.long	.LASF1205
	.byte	0x3c
	.byte	0x67
	.long	0x9726
	.uleb128 0x2d
	.long	.LASF1141
	.byte	0x3c
	.byte	0x6a
	.long	0x9efe
	.uleb128 0x2d
	.long	.LASF1206
	.byte	0x3c
	.byte	0x6b
	.long	0x9726
	.uleb128 0x2d
	.long	.LASF1207
	.byte	0x3c
	.byte	0x6c
	.long	0x9726
	.uleb128 0x20
	.long	.LASF1143
	.long	0x29
	.byte	0
	.uleb128 0x7
	.long	.LASF1209
	.byte	0x1
	.byte	0x3c
	.byte	0x64
	.long	0x81d4
	.uleb128 0x2d
	.long	.LASF1205
	.byte	0x3c
	.byte	0x67
	.long	0x9726
	.uleb128 0x2d
	.long	.LASF1141
	.byte	0x3c
	.byte	0x6a
	.long	0x9efe
	.uleb128 0x2d
	.long	.LASF1206
	.byte	0x3c
	.byte	0x6b
	.long	0x9726
	.uleb128 0x2d
	.long	.LASF1207
	.byte	0x3c
	.byte	0x6c
	.long	0x9726
	.uleb128 0x20
	.long	.LASF1143
	.long	0x9e79
	.byte	0
	.uleb128 0x7
	.long	.LASF1210
	.byte	0x1
	.byte	0x3c
	.byte	0x37
	.long	0x8216
	.uleb128 0x2d
	.long	.LASF1139
	.byte	0x3c
	.byte	0x3a
	.long	0x9f0f
	.uleb128 0x2d
	.long	.LASF1140
	.byte	0x3c
	.byte	0x3b
	.long	0x9f0f
	.uleb128 0x2d
	.long	.LASF1141
	.byte	0x3c
	.byte	0x3f
	.long	0x9efe
	.uleb128 0x2d
	.long	.LASF1142
	.byte	0x3c
	.byte	0x40
	.long	0x9726
	.uleb128 0x20
	.long	.LASF1143
	.long	0x9126
	.byte	0
	.uleb128 0x7
	.long	.LASF1211
	.byte	0x1
	.byte	0x3c
	.byte	0x37
	.long	0x8258
	.uleb128 0x2d
	.long	.LASF1139
	.byte	0x3c
	.byte	0x3a
	.long	0x9478
	.uleb128 0x2d
	.long	.LASF1140
	.byte	0x3c
	.byte	0x3b
	.long	0x9478
	.uleb128 0x2d
	.long	.LASF1141
	.byte	0x3c
	.byte	0x3f
	.long	0x9efe
	.uleb128 0x2d
	.long	.LASF1142
	.byte	0x3c
	.byte	0x40
	.long	0x9726
	.uleb128 0x20
	.long	.LASF1143
	.long	0x91a2
	.byte	0
	.uleb128 0x7
	.long	.LASF1212
	.byte	0x1
	.byte	0x3c
	.byte	0x37
	.long	0x829a
	.uleb128 0x2d
	.long	.LASF1139
	.byte	0x3c
	.byte	0x3a
	.long	0xa6d1
	.uleb128 0x2d
	.long	.LASF1140
	.byte	0x3c
	.byte	0x3b
	.long	0xa6d1
	.uleb128 0x2d
	.long	.LASF1141
	.byte	0x3c
	.byte	0x3f
	.long	0x9efe
	.uleb128 0x2d
	.long	.LASF1142
	.byte	0x3c
	.byte	0x40
	.long	0x9726
	.uleb128 0x20
	.long	.LASF1143
	.long	0x9149
	.byte	0
	.uleb128 0x7
	.long	.LASF1213
	.byte	0x1
	.byte	0x3c
	.byte	0x37
	.long	0x82dc
	.uleb128 0x2d
	.long	.LASF1139
	.byte	0x3c
	.byte	0x3a
	.long	0xa6d6
	.uleb128 0x2d
	.long	.LASF1140
	.byte	0x3c
	.byte	0x3b
	.long	0xa6d6
	.uleb128 0x2d
	.long	.LASF1141
	.byte	0x3c
	.byte	0x3f
	.long	0x9efe
	.uleb128 0x2d
	.long	.LASF1142
	.byte	0x3c
	.byte	0x40
	.long	0x9726
	.uleb128 0x20
	.long	.LASF1143
	.long	0x915b
	.byte	0
	.uleb128 0x7
	.long	.LASF1214
	.byte	0x1
	.byte	0x3d
	.byte	0x5f
	.long	0x83ec
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x2e5c
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x2e80
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x2ea0
	.uleb128 0x8
	.long	0x2df4
	.byte	0
	.uleb128 0x16
	.long	.LASF290
	.byte	0x3d
	.byte	0x67
	.long	0x2e0d
	.uleb128 0x16
	.long	.LASF4
	.byte	0x3d
	.byte	0x68
	.long	0x2e19
	.uleb128 0x16
	.long	.LASF10
	.byte	0x3d
	.byte	0x6d
	.long	0xb316
	.uleb128 0x16
	.long	.LASF11
	.byte	0x3d
	.byte	0x6e
	.long	0xb31c
	.uleb128 0x14
	.long	0x8303
	.uleb128 0x57
	.long	.LASF1146
	.byte	0x3d
	.byte	0x8b
	.long	.LASF1215
	.long	0x2eef
	.long	0x834d
	.uleb128 0xc
	.long	0xb322
	.byte	0
	.uleb128 0x30
	.long	.LASF1148
	.byte	0x3d
	.byte	0x8e
	.long	.LASF1216
	.long	0x8367
	.uleb128 0xc
	.long	0xb328
	.uleb128 0xc
	.long	0xb328
	.byte	0
	.uleb128 0x5e
	.long	.LASF1150
	.byte	0x3d
	.byte	0x91
	.long	.LASF1217
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1151
	.byte	0x3d
	.byte	0x94
	.long	.LASF1218
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1154
	.byte	0x3d
	.byte	0x97
	.long	.LASF1219
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1156
	.byte	0x3d
	.byte	0x9a
	.long	.LASF1220
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1158
	.byte	0x3d
	.byte	0x9d
	.long	.LASF1221
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1160
	.byte	0x3d
	.byte	0xa0
	.long	.LASF1222
	.long	0x9ef1
	.uleb128 0x7
	.long	.LASF1223
	.byte	0x1
	.byte	0x3d
	.byte	0xa8
	.long	0x83e2
	.uleb128 0x16
	.long	.LASF1163
	.byte	0x3d
	.byte	0xa9
	.long	0x2ed9
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.byte	0
	.uleb128 0x20
	.long	.LASF261
	.long	0x2eef
	.byte	0
	.uleb128 0x7
	.long	.LASF1224
	.byte	0x1
	.byte	0x11
	.byte	0x3a
	.long	0x8539
	.uleb128 0x16
	.long	.LASF5
	.byte	0x11
	.byte	0x3d
	.long	0x21c6
	.uleb128 0x16
	.long	.LASF4
	.byte	0x11
	.byte	0x3f
	.long	0xab81
	.uleb128 0x16
	.long	.LASF12
	.byte	0x11
	.byte	0x40
	.long	0xb2ff
	.uleb128 0x16
	.long	.LASF10
	.byte	0x11
	.byte	0x41
	.long	0xb32e
	.uleb128 0x16
	.long	.LASF11
	.byte	0x11
	.byte	0x42
	.long	0xb334
	.uleb128 0xa
	.long	.LASF1127
	.byte	0x11
	.byte	0x4f
	.long	.LASF1225
	.long	0x8442
	.long	0x8448
	.uleb128 0xb
	.long	0xb33a
	.byte	0
	.uleb128 0xa
	.long	.LASF1127
	.byte	0x11
	.byte	0x51
	.long	.LASF1226
	.long	0x845b
	.long	0x8466
	.uleb128 0xb
	.long	0xb33a
	.uleb128 0xc
	.long	0xb340
	.byte	0
	.uleb128 0xa
	.long	.LASF1130
	.byte	0x11
	.byte	0x56
	.long	.LASF1227
	.long	0x8479
	.long	0x8484
	.uleb128 0xb
	.long	0xb33a
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x17
	.long	.LASF1132
	.byte	0x11
	.byte	0x59
	.long	.LASF1228
	.long	0x8403
	.long	0x849b
	.long	0x84a6
	.uleb128 0xb
	.long	0xb346
	.uleb128 0xc
	.long	0x8419
	.byte	0
	.uleb128 0x17
	.long	.LASF1132
	.byte	0x11
	.byte	0x5d
	.long	.LASF1229
	.long	0x840e
	.long	0x84bd
	.long	0x84c8
	.uleb128 0xb
	.long	0xb346
	.uleb128 0xc
	.long	0x8424
	.byte	0
	.uleb128 0x17
	.long	.LASF335
	.byte	0x11
	.byte	0x63
	.long	.LASF1230
	.long	0x8403
	.long	0x84df
	.long	0x84ef
	.uleb128 0xb
	.long	0xb33a
	.uleb128 0xc
	.long	0x83f8
	.uleb128 0xc
	.long	0xa1f5
	.byte	0
	.uleb128 0xa
	.long	.LASF338
	.byte	0x11
	.byte	0x6d
	.long	.LASF1231
	.long	0x8502
	.long	0x8512
	.uleb128 0xb
	.long	0xb33a
	.uleb128 0xc
	.long	0x8403
	.uleb128 0xc
	.long	0x83f8
	.byte	0
	.uleb128 0x17
	.long	.LASF119
	.byte	0x11
	.byte	0x71
	.long	.LASF1232
	.long	0x83f8
	.long	0x8529
	.long	0x852f
	.uleb128 0xb
	.long	0xb346
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.byte	0
	.uleb128 0x14
	.long	0x83ec
	.uleb128 0x54
	.long	.LASF1233
	.byte	0x8
	.byte	0x3e
	.value	0x2d1
	.long	0x8765
	.uleb128 0x51
	.long	.LASF1165
	.byte	0x3e
	.value	0x2d4
	.long	0xab81
	.byte	0
	.byte	0x2
	.uleb128 0x42
	.long	.LASF619
	.byte	0x3e
	.value	0x2dc
	.long	0x3cf8
	.byte	0x1
	.uleb128 0x42
	.long	.LASF10
	.byte	0x3e
	.value	0x2dd
	.long	0x3d0e
	.byte	0x1
	.uleb128 0x42
	.long	.LASF4
	.byte	0x3e
	.value	0x2de
	.long	0x3d03
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF1166
	.byte	0x3e
	.value	0x2e0
	.long	.LASF1234
	.byte	0x1
	.long	0x8595
	.long	0x859b
	.uleb128 0xb
	.long	0xb3c4
	.byte	0
	.uleb128 0x1d
	.long	.LASF1166
	.byte	0x3e
	.value	0x2e4
	.long	.LASF1235
	.byte	0x1
	.long	0x85b0
	.long	0x85bb
	.uleb128 0xb
	.long	0xb3c4
	.uleb128 0xc
	.long	0xb3ca
	.byte	0
	.uleb128 0x1e
	.long	.LASF1169
	.byte	0x3e
	.value	0x2f1
	.long	.LASF1236
	.long	0x8566
	.byte	0x1
	.long	0x85d4
	.long	0x85da
	.uleb128 0xb
	.long	0xb3d5
	.byte	0
	.uleb128 0x1e
	.long	.LASF1171
	.byte	0x3e
	.value	0x2f5
	.long	.LASF1237
	.long	0x8573
	.byte	0x1
	.long	0x85f3
	.long	0x85f9
	.uleb128 0xb
	.long	0xb3d5
	.byte	0
	.uleb128 0x1e
	.long	.LASF1173
	.byte	0x3e
	.value	0x2f9
	.long	.LASF1238
	.long	0xb3db
	.byte	0x1
	.long	0x8612
	.long	0x8618
	.uleb128 0xb
	.long	0xb3c4
	.byte	0
	.uleb128 0x1e
	.long	.LASF1173
	.byte	0x3e
	.value	0x300
	.long	.LASF1239
	.long	0x853e
	.byte	0x1
	.long	0x8631
	.long	0x863c
	.uleb128 0xb
	.long	0xb3c4
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1176
	.byte	0x3e
	.value	0x305
	.long	.LASF1240
	.long	0xb3db
	.byte	0x1
	.long	0x8655
	.long	0x865b
	.uleb128 0xb
	.long	0xb3c4
	.byte	0
	.uleb128 0x1e
	.long	.LASF1176
	.byte	0x3e
	.value	0x30c
	.long	.LASF1241
	.long	0x853e
	.byte	0x1
	.long	0x8674
	.long	0x867f
	.uleb128 0xb
	.long	0xb3c4
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x3e
	.value	0x311
	.long	.LASF1242
	.long	0x8566
	.byte	0x1
	.long	0x8698
	.long	0x86a3
	.uleb128 0xb
	.long	0xb3d5
	.uleb128 0xc
	.long	0x8559
	.byte	0
	.uleb128 0x1e
	.long	.LASF145
	.byte	0x3e
	.value	0x315
	.long	.LASF1243
	.long	0xb3db
	.byte	0x1
	.long	0x86bc
	.long	0x86c7
	.uleb128 0xb
	.long	0xb3c4
	.uleb128 0xc
	.long	0x8559
	.byte	0
	.uleb128 0x1e
	.long	.LASF1181
	.byte	0x3e
	.value	0x319
	.long	.LASF1244
	.long	0x853e
	.byte	0x1
	.long	0x86e0
	.long	0x86eb
	.uleb128 0xb
	.long	0xb3d5
	.uleb128 0xc
	.long	0x8559
	.byte	0
	.uleb128 0x1e
	.long	.LASF637
	.byte	0x3e
	.value	0x31d
	.long	.LASF1245
	.long	0xb3db
	.byte	0x1
	.long	0x8704
	.long	0x870f
	.uleb128 0xb
	.long	0xb3c4
	.uleb128 0xc
	.long	0x8559
	.byte	0
	.uleb128 0x1e
	.long	.LASF1184
	.byte	0x3e
	.value	0x321
	.long	.LASF1246
	.long	0x853e
	.byte	0x1
	.long	0x8728
	.long	0x8733
	.uleb128 0xb
	.long	0xb3d5
	.uleb128 0xc
	.long	0x8559
	.byte	0
	.uleb128 0x1e
	.long	.LASF1186
	.byte	0x3e
	.value	0x325
	.long	.LASF1247
	.long	0xb3ca
	.byte	0x1
	.long	0x874c
	.long	0x8752
	.uleb128 0xb
	.long	0xb3d5
	.byte	0
	.uleb128 0x20
	.long	.LASF620
	.long	0xab81
	.uleb128 0x20
	.long	.LASF1188
	.long	0x320d
	.byte	0
	.uleb128 0x2a
	.long	.LASF1248
	.uleb128 0x14
	.long	0x853e
	.uleb128 0x7
	.long	.LASF1249
	.byte	0x1
	.byte	0x3d
	.byte	0x5f
	.long	0x887f
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x43f4
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x4418
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x4438
	.uleb128 0x8
	.long	0x438c
	.byte	0
	.uleb128 0x16
	.long	.LASF290
	.byte	0x3d
	.byte	0x67
	.long	0x43a5
	.uleb128 0x16
	.long	.LASF4
	.byte	0x3d
	.byte	0x68
	.long	0x43b1
	.uleb128 0x16
	.long	.LASF10
	.byte	0x3d
	.byte	0x6d
	.long	0x14145
	.uleb128 0x16
	.long	.LASF11
	.byte	0x3d
	.byte	0x6e
	.long	0x1414b
	.uleb128 0x14
	.long	0x8796
	.uleb128 0x57
	.long	.LASF1146
	.byte	0x3d
	.byte	0x8b
	.long	.LASF1250
	.long	0x4487
	.long	0x87e0
	.uleb128 0xc
	.long	0x14151
	.byte	0
	.uleb128 0x30
	.long	.LASF1148
	.byte	0x3d
	.byte	0x8e
	.long	.LASF1251
	.long	0x87fa
	.uleb128 0xc
	.long	0x14157
	.uleb128 0xc
	.long	0x14157
	.byte	0
	.uleb128 0x5e
	.long	.LASF1150
	.byte	0x3d
	.byte	0x91
	.long	.LASF1252
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1151
	.byte	0x3d
	.byte	0x94
	.long	.LASF1253
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1154
	.byte	0x3d
	.byte	0x97
	.long	.LASF1254
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1156
	.byte	0x3d
	.byte	0x9a
	.long	.LASF1255
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1158
	.byte	0x3d
	.byte	0x9d
	.long	.LASF1256
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1160
	.byte	0x3d
	.byte	0xa0
	.long	.LASF1257
	.long	0x9ef1
	.uleb128 0x7
	.long	.LASF1258
	.byte	0x1
	.byte	0x3d
	.byte	0xa8
	.long	0x8875
	.uleb128 0x16
	.long	.LASF1163
	.byte	0x3d
	.byte	0xa9
	.long	0x4471
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x912d
	.byte	0
	.uleb128 0x20
	.long	.LASF261
	.long	0x4487
	.byte	0
	.uleb128 0x7
	.long	.LASF1259
	.byte	0x1
	.byte	0x11
	.byte	0x3a
	.long	0x89cc
	.uleb128 0x16
	.long	.LASF5
	.byte	0x11
	.byte	0x3d
	.long	0x21c6
	.uleb128 0x16
	.long	.LASF4
	.byte	0x11
	.byte	0x3f
	.long	0x14128
	.uleb128 0x16
	.long	.LASF12
	.byte	0x11
	.byte	0x40
	.long	0x1412e
	.uleb128 0x16
	.long	.LASF10
	.byte	0x11
	.byte	0x41
	.long	0x1415d
	.uleb128 0x16
	.long	.LASF11
	.byte	0x11
	.byte	0x42
	.long	0x14163
	.uleb128 0xa
	.long	.LASF1127
	.byte	0x11
	.byte	0x4f
	.long	.LASF1260
	.long	0x88d5
	.long	0x88db
	.uleb128 0xb
	.long	0x14169
	.byte	0
	.uleb128 0xa
	.long	.LASF1127
	.byte	0x11
	.byte	0x51
	.long	.LASF1261
	.long	0x88ee
	.long	0x88f9
	.uleb128 0xb
	.long	0x14169
	.uleb128 0xc
	.long	0x1416f
	.byte	0
	.uleb128 0xa
	.long	.LASF1130
	.byte	0x11
	.byte	0x56
	.long	.LASF1262
	.long	0x890c
	.long	0x8917
	.uleb128 0xb
	.long	0x14169
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x17
	.long	.LASF1132
	.byte	0x11
	.byte	0x59
	.long	.LASF1263
	.long	0x8896
	.long	0x892e
	.long	0x8939
	.uleb128 0xb
	.long	0x14175
	.uleb128 0xc
	.long	0x88ac
	.byte	0
	.uleb128 0x17
	.long	.LASF1132
	.byte	0x11
	.byte	0x5d
	.long	.LASF1264
	.long	0x88a1
	.long	0x8950
	.long	0x895b
	.uleb128 0xb
	.long	0x14175
	.uleb128 0xc
	.long	0x88b7
	.byte	0
	.uleb128 0x17
	.long	.LASF335
	.byte	0x11
	.byte	0x63
	.long	.LASF1265
	.long	0x8896
	.long	0x8972
	.long	0x8982
	.uleb128 0xb
	.long	0x14169
	.uleb128 0xc
	.long	0x888b
	.uleb128 0xc
	.long	0xa1f5
	.byte	0
	.uleb128 0xa
	.long	.LASF338
	.byte	0x11
	.byte	0x6d
	.long	.LASF1266
	.long	0x8995
	.long	0x89a5
	.uleb128 0xb
	.long	0x14169
	.uleb128 0xc
	.long	0x8896
	.uleb128 0xc
	.long	0x888b
	.byte	0
	.uleb128 0x17
	.long	.LASF119
	.byte	0x11
	.byte	0x71
	.long	.LASF1267
	.long	0x888b
	.long	0x89bc
	.long	0x89c2
	.uleb128 0xb
	.long	0x14175
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x912d
	.byte	0
	.uleb128 0x14
	.long	0x887f
	.uleb128 0x2a
	.long	.LASF1268
	.uleb128 0x2a
	.long	.LASF1269
	.uleb128 0x7
	.long	.LASF1270
	.byte	0x1
	.byte	0x3d
	.byte	0x5f
	.long	0x8aeb
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x5303
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x5327
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x5347
	.uleb128 0x8
	.long	0x529b
	.byte	0
	.uleb128 0x16
	.long	.LASF290
	.byte	0x3d
	.byte	0x67
	.long	0x52b4
	.uleb128 0x16
	.long	.LASF4
	.byte	0x3d
	.byte	0x68
	.long	0x52c0
	.uleb128 0x16
	.long	.LASF10
	.byte	0x3d
	.byte	0x6d
	.long	0x141f9
	.uleb128 0x16
	.long	.LASF11
	.byte	0x3d
	.byte	0x6e
	.long	0x141ff
	.uleb128 0x14
	.long	0x8a02
	.uleb128 0x57
	.long	.LASF1146
	.byte	0x3d
	.byte	0x8b
	.long	.LASF1271
	.long	0x5396
	.long	0x8a4c
	.uleb128 0xc
	.long	0x14205
	.byte	0
	.uleb128 0x30
	.long	.LASF1148
	.byte	0x3d
	.byte	0x8e
	.long	.LASF1272
	.long	0x8a66
	.uleb128 0xc
	.long	0x1420b
	.uleb128 0xc
	.long	0x1420b
	.byte	0
	.uleb128 0x5e
	.long	.LASF1150
	.byte	0x3d
	.byte	0x91
	.long	.LASF1273
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1151
	.byte	0x3d
	.byte	0x94
	.long	.LASF1274
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1154
	.byte	0x3d
	.byte	0x97
	.long	.LASF1275
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1156
	.byte	0x3d
	.byte	0x9a
	.long	.LASF1276
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1158
	.byte	0x3d
	.byte	0x9d
	.long	.LASF1277
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1160
	.byte	0x3d
	.byte	0xa0
	.long	.LASF1278
	.long	0x9ef1
	.uleb128 0x7
	.long	.LASF1279
	.byte	0x1
	.byte	0x3d
	.byte	0xa8
	.long	0x8ae1
	.uleb128 0x16
	.long	.LASF1163
	.byte	0x3d
	.byte	0xa9
	.long	0x5380
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x30
	.byte	0
	.uleb128 0x20
	.long	.LASF261
	.long	0x5396
	.byte	0
	.uleb128 0x7
	.long	.LASF1280
	.byte	0x1
	.byte	0x11
	.byte	0x3a
	.long	0x8c38
	.uleb128 0x16
	.long	.LASF5
	.byte	0x11
	.byte	0x3d
	.long	0x21c6
	.uleb128 0x16
	.long	.LASF4
	.byte	0x11
	.byte	0x3f
	.long	0xaa8a
	.uleb128 0x16
	.long	.LASF12
	.byte	0x11
	.byte	0x40
	.long	0x9720
	.uleb128 0x16
	.long	.LASF10
	.byte	0x11
	.byte	0x41
	.long	0x13fc2
	.uleb128 0x16
	.long	.LASF11
	.byte	0x11
	.byte	0x42
	.long	0x13fbc
	.uleb128 0xa
	.long	.LASF1127
	.byte	0x11
	.byte	0x4f
	.long	.LASF1281
	.long	0x8b41
	.long	0x8b47
	.uleb128 0xb
	.long	0x14211
	.byte	0
	.uleb128 0xa
	.long	.LASF1127
	.byte	0x11
	.byte	0x51
	.long	.LASF1282
	.long	0x8b5a
	.long	0x8b65
	.uleb128 0xb
	.long	0x14211
	.uleb128 0xc
	.long	0x14217
	.byte	0
	.uleb128 0xa
	.long	.LASF1130
	.byte	0x11
	.byte	0x56
	.long	.LASF1283
	.long	0x8b78
	.long	0x8b83
	.uleb128 0xb
	.long	0x14211
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x17
	.long	.LASF1132
	.byte	0x11
	.byte	0x59
	.long	.LASF1284
	.long	0x8b02
	.long	0x8b9a
	.long	0x8ba5
	.uleb128 0xb
	.long	0x1421d
	.uleb128 0xc
	.long	0x8b18
	.byte	0
	.uleb128 0x17
	.long	.LASF1132
	.byte	0x11
	.byte	0x5d
	.long	.LASF1285
	.long	0x8b0d
	.long	0x8bbc
	.long	0x8bc7
	.uleb128 0xb
	.long	0x1421d
	.uleb128 0xc
	.long	0x8b23
	.byte	0
	.uleb128 0x17
	.long	.LASF335
	.byte	0x11
	.byte	0x63
	.long	.LASF1286
	.long	0x8b02
	.long	0x8bde
	.long	0x8bee
	.uleb128 0xb
	.long	0x14211
	.uleb128 0xc
	.long	0x8af7
	.uleb128 0xc
	.long	0xa1f5
	.byte	0
	.uleb128 0xa
	.long	.LASF338
	.byte	0x11
	.byte	0x6d
	.long	.LASF1287
	.long	0x8c01
	.long	0x8c11
	.uleb128 0xb
	.long	0x14211
	.uleb128 0xc
	.long	0x8b02
	.uleb128 0xc
	.long	0x8af7
	.byte	0
	.uleb128 0x17
	.long	.LASF119
	.byte	0x11
	.byte	0x71
	.long	.LASF1288
	.long	0x8af7
	.long	0x8c28
	.long	0x8c2e
	.uleb128 0xb
	.long	0x1421d
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x30
	.byte	0
	.uleb128 0x14
	.long	0x8aeb
	.uleb128 0x2a
	.long	.LASF1289
	.uleb128 0x54
	.long	.LASF1290
	.byte	0x8
	.byte	0x3e
	.value	0x2d1
	.long	0x8e69
	.uleb128 0x51
	.long	.LASF1165
	.byte	0x3e
	.value	0x2d4
	.long	0x9720
	.byte	0
	.byte	0x2
	.uleb128 0x42
	.long	.LASF619
	.byte	0x3e
	.value	0x2dc
	.long	0x723a
	.byte	0x1
	.uleb128 0x42
	.long	.LASF10
	.byte	0x3e
	.value	0x2dd
	.long	0x7250
	.byte	0x1
	.uleb128 0x42
	.long	.LASF4
	.byte	0x3e
	.value	0x2de
	.long	0x7245
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF1166
	.byte	0x3e
	.value	0x2e0
	.long	.LASF1291
	.byte	0x1
	.long	0x8c99
	.long	0x8c9f
	.uleb128 0xb
	.long	0x1495d
	.byte	0
	.uleb128 0x1d
	.long	.LASF1166
	.byte	0x3e
	.value	0x2e4
	.long	.LASF1292
	.byte	0x1
	.long	0x8cb4
	.long	0x8cbf
	.uleb128 0xb
	.long	0x1495d
	.uleb128 0xc
	.long	0x14963
	.byte	0
	.uleb128 0x1e
	.long	.LASF1169
	.byte	0x3e
	.value	0x2f1
	.long	.LASF1293
	.long	0x8c6a
	.byte	0x1
	.long	0x8cd8
	.long	0x8cde
	.uleb128 0xb
	.long	0x1496e
	.byte	0
	.uleb128 0x1e
	.long	.LASF1171
	.byte	0x3e
	.value	0x2f5
	.long	.LASF1294
	.long	0x8c77
	.byte	0x1
	.long	0x8cf7
	.long	0x8cfd
	.uleb128 0xb
	.long	0x1496e
	.byte	0
	.uleb128 0x1e
	.long	.LASF1173
	.byte	0x3e
	.value	0x2f9
	.long	.LASF1295
	.long	0x14974
	.byte	0x1
	.long	0x8d16
	.long	0x8d1c
	.uleb128 0xb
	.long	0x1495d
	.byte	0
	.uleb128 0x1e
	.long	.LASF1173
	.byte	0x3e
	.value	0x300
	.long	.LASF1296
	.long	0x8c42
	.byte	0x1
	.long	0x8d35
	.long	0x8d40
	.uleb128 0xb
	.long	0x1495d
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1176
	.byte	0x3e
	.value	0x305
	.long	.LASF1297
	.long	0x14974
	.byte	0x1
	.long	0x8d59
	.long	0x8d5f
	.uleb128 0xb
	.long	0x1495d
	.byte	0
	.uleb128 0x1e
	.long	.LASF1176
	.byte	0x3e
	.value	0x30c
	.long	.LASF1298
	.long	0x8c42
	.byte	0x1
	.long	0x8d78
	.long	0x8d83
	.uleb128 0xb
	.long	0x1495d
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x3e
	.value	0x311
	.long	.LASF1299
	.long	0x8c6a
	.byte	0x1
	.long	0x8d9c
	.long	0x8da7
	.uleb128 0xb
	.long	0x1496e
	.uleb128 0xc
	.long	0x8c5d
	.byte	0
	.uleb128 0x1e
	.long	.LASF145
	.byte	0x3e
	.value	0x315
	.long	.LASF1300
	.long	0x14974
	.byte	0x1
	.long	0x8dc0
	.long	0x8dcb
	.uleb128 0xb
	.long	0x1495d
	.uleb128 0xc
	.long	0x8c5d
	.byte	0
	.uleb128 0x1e
	.long	.LASF1181
	.byte	0x3e
	.value	0x319
	.long	.LASF1301
	.long	0x8c42
	.byte	0x1
	.long	0x8de4
	.long	0x8def
	.uleb128 0xb
	.long	0x1496e
	.uleb128 0xc
	.long	0x8c5d
	.byte	0
	.uleb128 0x1e
	.long	.LASF637
	.byte	0x3e
	.value	0x31d
	.long	.LASF1302
	.long	0x14974
	.byte	0x1
	.long	0x8e08
	.long	0x8e13
	.uleb128 0xb
	.long	0x1495d
	.uleb128 0xc
	.long	0x8c5d
	.byte	0
	.uleb128 0x1e
	.long	.LASF1184
	.byte	0x3e
	.value	0x321
	.long	.LASF1303
	.long	0x8c42
	.byte	0x1
	.long	0x8e2c
	.long	0x8e37
	.uleb128 0xb
	.long	0x1496e
	.uleb128 0xc
	.long	0x8c5d
	.byte	0
	.uleb128 0x1e
	.long	.LASF1186
	.byte	0x3e
	.value	0x325
	.long	.LASF1304
	.long	0x14963
	.byte	0x1
	.long	0x8e50
	.long	0x8e56
	.uleb128 0xb
	.long	0x1496e
	.byte	0
	.uleb128 0x20
	.long	.LASF620
	.long	0x9720
	.uleb128 0x20
	.long	.LASF1188
	.long	0x56d0
	.byte	0
	.uleb128 0x7
	.long	.LASF1305
	.byte	0x1
	.byte	0x3d
	.byte	0x5f
	.long	0x8f79
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x6bfa
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x6c1e
	.uleb128 0x23
	.byte	0x3d
	.byte	0x5f
	.long	0x6c3e
	.uleb128 0x8
	.long	0x6b92
	.byte	0
	.uleb128 0x16
	.long	.LASF290
	.byte	0x3d
	.byte	0x67
	.long	0x6bab
	.uleb128 0x16
	.long	.LASF4
	.byte	0x3d
	.byte	0x68
	.long	0x6bb7
	.uleb128 0x16
	.long	.LASF10
	.byte	0x3d
	.byte	0x6d
	.long	0x143f7
	.uleb128 0x16
	.long	.LASF11
	.byte	0x3d
	.byte	0x6e
	.long	0x143fd
	.uleb128 0x14
	.long	0x8e90
	.uleb128 0x57
	.long	.LASF1146
	.byte	0x3d
	.byte	0x8b
	.long	.LASF1306
	.long	0x6c8d
	.long	0x8eda
	.uleb128 0xc
	.long	0x14403
	.byte	0
	.uleb128 0x30
	.long	.LASF1148
	.byte	0x3d
	.byte	0x8e
	.long	.LASF1307
	.long	0x8ef4
	.uleb128 0xc
	.long	0x14409
	.uleb128 0xc
	.long	0x14409
	.byte	0
	.uleb128 0x5e
	.long	.LASF1150
	.byte	0x3d
	.byte	0x91
	.long	.LASF1308
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1151
	.byte	0x3d
	.byte	0x94
	.long	.LASF1309
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1154
	.byte	0x3d
	.byte	0x97
	.long	.LASF1310
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1156
	.byte	0x3d
	.byte	0x9a
	.long	.LASF1311
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1158
	.byte	0x3d
	.byte	0x9d
	.long	.LASF1312
	.long	0x9ef1
	.uleb128 0x5e
	.long	.LASF1160
	.byte	0x3d
	.byte	0xa0
	.long	.LASF1313
	.long	0x9ef1
	.uleb128 0x7
	.long	.LASF1314
	.byte	0x1
	.byte	0x3d
	.byte	0xa8
	.long	0x8f6f
	.uleb128 0x16
	.long	.LASF1163
	.byte	0x3d
	.byte	0xa9
	.long	0x6c77
	.uleb128 0x2c
	.string	"_Tp"
	.long	0xc3f6
	.byte	0
	.uleb128 0x20
	.long	.LASF261
	.long	0x6c8d
	.byte	0
	.uleb128 0x7
	.long	.LASF1315
	.byte	0x1
	.byte	0x11
	.byte	0x3a
	.long	0x90c6
	.uleb128 0x16
	.long	.LASF5
	.byte	0x11
	.byte	0x3d
	.long	0x21c6
	.uleb128 0x16
	.long	.LASF4
	.byte	0x11
	.byte	0x3f
	.long	0x14013
	.uleb128 0x16
	.long	.LASF12
	.byte	0x11
	.byte	0x40
	.long	0x1406a
	.uleb128 0x16
	.long	.LASF10
	.byte	0x11
	.byte	0x41
	.long	0x1405e
	.uleb128 0x16
	.long	.LASF11
	.byte	0x11
	.byte	0x42
	.long	0x1401f
	.uleb128 0xa
	.long	.LASF1127
	.byte	0x11
	.byte	0x4f
	.long	.LASF1316
	.long	0x8fcf
	.long	0x8fd5
	.uleb128 0xb
	.long	0x1440f
	.byte	0
	.uleb128 0xa
	.long	.LASF1127
	.byte	0x11
	.byte	0x51
	.long	.LASF1317
	.long	0x8fe8
	.long	0x8ff3
	.uleb128 0xb
	.long	0x1440f
	.uleb128 0xc
	.long	0x14415
	.byte	0
	.uleb128 0xa
	.long	.LASF1130
	.byte	0x11
	.byte	0x56
	.long	.LASF1318
	.long	0x9006
	.long	0x9011
	.uleb128 0xb
	.long	0x1440f
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x17
	.long	.LASF1132
	.byte	0x11
	.byte	0x59
	.long	.LASF1319
	.long	0x8f90
	.long	0x9028
	.long	0x9033
	.uleb128 0xb
	.long	0x1441b
	.uleb128 0xc
	.long	0x8fa6
	.byte	0
	.uleb128 0x17
	.long	.LASF1132
	.byte	0x11
	.byte	0x5d
	.long	.LASF1320
	.long	0x8f9b
	.long	0x904a
	.long	0x9055
	.uleb128 0xb
	.long	0x1441b
	.uleb128 0xc
	.long	0x8fb1
	.byte	0
	.uleb128 0x17
	.long	.LASF335
	.byte	0x11
	.byte	0x63
	.long	.LASF1321
	.long	0x8f90
	.long	0x906c
	.long	0x907c
	.uleb128 0xb
	.long	0x1440f
	.uleb128 0xc
	.long	0x8f85
	.uleb128 0xc
	.long	0xa1f5
	.byte	0
	.uleb128 0xa
	.long	.LASF338
	.byte	0x11
	.byte	0x6d
	.long	.LASF1322
	.long	0x908f
	.long	0x909f
	.uleb128 0xb
	.long	0x1440f
	.uleb128 0xc
	.long	0x8f90
	.uleb128 0xc
	.long	0x8f85
	.byte	0
	.uleb128 0x17
	.long	.LASF119
	.byte	0x11
	.byte	0x71
	.long	.LASF1323
	.long	0x8f85
	.long	0x90b6
	.long	0x90bc
	.uleb128 0xb
	.long	0x1441b
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0xc3f6
	.byte	0
	.uleb128 0x14
	.long	0x8f79
	.uleb128 0x2a
	.long	.LASF1324
	.uleb128 0x2a
	.long	.LASF1325
	.uleb128 0x14
	.long	0x7ee7
	.uleb128 0x14
	.long	0x7cc0
	.uleb128 0x14
	.long	0x8c42
	.uleb128 0x7
	.long	.LASF1326
	.byte	0x1
	.byte	0x3f
	.byte	0x2f
	.long	0x90fc
	.uleb128 0x16
	.long	.LASF1012
	.byte	0x3f
	.byte	0x30
	.long	0x14128
	.byte	0
	.uleb128 0x5f
	.long	.LASF1327
	.byte	0x3f
	.byte	0x96
	.long	.LASF1328
	.long	0x9ef1
	.uleb128 0x20
	.long	.LASF1329
	.long	0x9478
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.byte	0
	.uleb128 0x16
	.long	.LASF323
	.byte	0x40
	.byte	0xd8
	.long	0x9126
	.uleb128 0x2
	.byte	0x8
	.byte	0x7
	.long	.LASF1330
	.uleb128 0x2
	.byte	0x1
	.byte	0x8
	.long	.LASF1331
	.uleb128 0x2
	.byte	0x2
	.byte	0x7
	.long	.LASF1332
	.uleb128 0x2
	.byte	0x4
	.byte	0x7
	.long	.LASF1333
	.uleb128 0x2
	.byte	0x1
	.byte	0x6
	.long	.LASF1334
	.uleb128 0x2
	.byte	0x2
	.byte	0x5
	.long	.LASF1335
	.uleb128 0x16
	.long	.LASF1336
	.byte	0x41
	.byte	0x28
	.long	0x30
	.uleb128 0x2
	.byte	0x8
	.byte	0x5
	.long	.LASF1337
	.uleb128 0x16
	.long	.LASF1338
	.byte	0x41
	.byte	0x83
	.long	0x915b
	.uleb128 0x16
	.long	.LASF1339
	.byte	0x41
	.byte	0x84
	.long	0x915b
	.uleb128 0x60
	.long	0x30
	.long	0x9188
	.uleb128 0x61
	.long	0x9188
	.byte	0x1
	.byte	0
	.uleb128 0x2
	.byte	0x8
	.byte	0x7
	.long	.LASF1340
	.uleb128 0x16
	.long	.LASF1341
	.byte	0x41
	.byte	0x87
	.long	0x915b
	.uleb128 0x62
	.byte	0x8
	.uleb128 0x63
	.byte	0x8
	.long	0x91a2
	.uleb128 0x2
	.byte	0x1
	.byte	0x6
	.long	.LASF1342
	.uleb128 0x16
	.long	.LASF1343
	.byte	0x42
	.byte	0x30
	.long	0x91b4
	.uleb128 0x7
	.long	.LASF1344
	.byte	0xd8
	.byte	0x43
	.byte	0xf1
	.long	0x9331
	.uleb128 0x9
	.long	.LASF1345
	.byte	0x43
	.byte	0xf2
	.long	0x30
	.byte	0
	.uleb128 0x9
	.long	.LASF1346
	.byte	0x43
	.byte	0xf7
	.long	0x919c
	.byte	0x8
	.uleb128 0x9
	.long	.LASF1347
	.byte	0x43
	.byte	0xf8
	.long	0x919c
	.byte	0x10
	.uleb128 0x9
	.long	.LASF1348
	.byte	0x43
	.byte	0xf9
	.long	0x919c
	.byte	0x18
	.uleb128 0x9
	.long	.LASF1349
	.byte	0x43
	.byte	0xfa
	.long	0x919c
	.byte	0x20
	.uleb128 0x9
	.long	.LASF1350
	.byte	0x43
	.byte	0xfb
	.long	0x919c
	.byte	0x28
	.uleb128 0x9
	.long	.LASF1351
	.byte	0x43
	.byte	0xfc
	.long	0x919c
	.byte	0x30
	.uleb128 0x9
	.long	.LASF1352
	.byte	0x43
	.byte	0xfd
	.long	0x919c
	.byte	0x38
	.uleb128 0x9
	.long	.LASF1353
	.byte	0x43
	.byte	0xfe
	.long	0x919c
	.byte	0x40
	.uleb128 0x4b
	.long	.LASF1354
	.byte	0x43
	.value	0x100
	.long	0x919c
	.byte	0x48
	.uleb128 0x4b
	.long	.LASF1355
	.byte	0x43
	.value	0x101
	.long	0x919c
	.byte	0x50
	.uleb128 0x4b
	.long	.LASF1356
	.byte	0x43
	.value	0x102
	.long	0x919c
	.byte	0x58
	.uleb128 0x4b
	.long	.LASF1357
	.byte	0x43
	.value	0x104
	.long	0x9440
	.byte	0x60
	.uleb128 0x4b
	.long	.LASF1358
	.byte	0x43
	.value	0x106
	.long	0x9446
	.byte	0x68
	.uleb128 0x4b
	.long	.LASF1359
	.byte	0x43
	.value	0x108
	.long	0x30
	.byte	0x70
	.uleb128 0x4b
	.long	.LASF1360
	.byte	0x43
	.value	0x10c
	.long	0x30
	.byte	0x74
	.uleb128 0x4b
	.long	.LASF1361
	.byte	0x43
	.value	0x10e
	.long	0x9162
	.byte	0x78
	.uleb128 0x4b
	.long	.LASF1362
	.byte	0x43
	.value	0x112
	.long	0x9134
	.byte	0x80
	.uleb128 0x4b
	.long	.LASF1363
	.byte	0x43
	.value	0x113
	.long	0x9142
	.byte	0x82
	.uleb128 0x4b
	.long	.LASF1364
	.byte	0x43
	.value	0x114
	.long	0x944c
	.byte	0x83
	.uleb128 0x4b
	.long	.LASF1365
	.byte	0x43
	.value	0x118
	.long	0x945c
	.byte	0x88
	.uleb128 0x4b
	.long	.LASF1366
	.byte	0x43
	.value	0x121
	.long	0x916d
	.byte	0x90
	.uleb128 0x4b
	.long	.LASF1367
	.byte	0x43
	.value	0x129
	.long	0x919a
	.byte	0x98
	.uleb128 0x4b
	.long	.LASF1368
	.byte	0x43
	.value	0x12a
	.long	0x919a
	.byte	0xa0
	.uleb128 0x4b
	.long	.LASF1369
	.byte	0x43
	.value	0x12b
	.long	0x919a
	.byte	0xa8
	.uleb128 0x4b
	.long	.LASF1370
	.byte	0x43
	.value	0x12c
	.long	0x919a
	.byte	0xb0
	.uleb128 0x4b
	.long	.LASF1371
	.byte	0x43
	.value	0x12e
	.long	0x911b
	.byte	0xb8
	.uleb128 0x4b
	.long	.LASF1372
	.byte	0x43
	.value	0x12f
	.long	0x30
	.byte	0xc0
	.uleb128 0x4b
	.long	.LASF1373
	.byte	0x43
	.value	0x131
	.long	0x9462
	.byte	0xc4
	.byte	0
	.uleb128 0x16
	.long	.LASF1374
	.byte	0x42
	.byte	0x40
	.long	0x91b4
	.uleb128 0x64
	.byte	0x8
	.byte	0x44
	.byte	0x53
	.long	.LASF1379
	.long	0x9380
	.uleb128 0x11
	.byte	0x4
	.byte	0x44
	.byte	0x56
	.long	0x9367
	.uleb128 0x12
	.long	.LASF1375
	.byte	0x44
	.byte	0x58
	.long	0x913b
	.uleb128 0x12
	.long	.LASF1376
	.byte	0x44
	.byte	0x5c
	.long	0x9380
	.byte	0
	.uleb128 0x9
	.long	.LASF1377
	.byte	0x44
	.byte	0x54
	.long	0x30
	.byte	0
	.uleb128 0x9
	.long	.LASF288
	.byte	0x44
	.byte	0x5d
	.long	0x9348
	.byte	0x4
	.byte	0
	.uleb128 0x60
	.long	0x91a2
	.long	0x9390
	.uleb128 0x61
	.long	0x9188
	.byte	0x3
	.byte	0
	.uleb128 0x16
	.long	.LASF1378
	.byte	0x44
	.byte	0x5e
	.long	0x933c
	.uleb128 0x64
	.byte	0x10
	.byte	0x45
	.byte	0x16
	.long	.LASF1380
	.long	0x93c0
	.uleb128 0x9
	.long	.LASF1381
	.byte	0x45
	.byte	0x17
	.long	0x9162
	.byte	0
	.uleb128 0x9
	.long	.LASF1382
	.byte	0x45
	.byte	0x18
	.long	0x9390
	.byte	0x8
	.byte	0
	.uleb128 0x16
	.long	.LASF1383
	.byte	0x45
	.byte	0x19
	.long	0x939b
	.uleb128 0x7
	.long	.LASF1384
	.byte	0x18
	.byte	0x46
	.byte	0
	.long	0x9408
	.uleb128 0x9
	.long	.LASF1385
	.byte	0x46
	.byte	0
	.long	0x913b
	.byte	0
	.uleb128 0x9
	.long	.LASF1386
	.byte	0x46
	.byte	0
	.long	0x913b
	.byte	0x4
	.uleb128 0x9
	.long	.LASF1387
	.byte	0x46
	.byte	0
	.long	0x919a
	.byte	0x8
	.uleb128 0x9
	.long	.LASF1388
	.byte	0x46
	.byte	0
	.long	0x919a
	.byte	0x10
	.byte	0
	.uleb128 0x65
	.long	.LASF3073
	.byte	0x43
	.byte	0x96
	.uleb128 0x7
	.long	.LASF1389
	.byte	0x18
	.byte	0x43
	.byte	0x9c
	.long	0x9440
	.uleb128 0x9
	.long	.LASF1390
	.byte	0x43
	.byte	0x9d
	.long	0x9440
	.byte	0
	.uleb128 0x9
	.long	.LASF1391
	.byte	0x43
	.byte	0x9e
	.long	0x9446
	.byte	0x8
	.uleb128 0x9
	.long	.LASF1392
	.byte	0x43
	.byte	0xa2
	.long	0x30
	.byte	0x10
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x940f
	.uleb128 0x63
	.byte	0x8
	.long	0x91b4
	.uleb128 0x60
	.long	0x91a2
	.long	0x945c
	.uleb128 0x61
	.long	0x9188
	.byte	0
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x9408
	.uleb128 0x60
	.long	0x91a2
	.long	0x9472
	.uleb128 0x61
	.long	0x9188
	.byte	0x13
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x9478
	.uleb128 0x14
	.long	0x91a2
	.uleb128 0x16
	.long	.LASF1393
	.byte	0x42
	.byte	0x6e
	.long	0x93c0
	.uleb128 0x66
	.long	.LASF1394
	.byte	0x42
	.value	0x33a
	.long	0x949a
	.uleb128 0xc
	.long	0x949a
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x91a9
	.uleb128 0x67
	.long	.LASF1395
	.byte	0x42
	.byte	0xed
	.long	0x30
	.long	0x94b5
	.uleb128 0xc
	.long	0x949a
	.byte	0
	.uleb128 0x68
	.long	.LASF1396
	.byte	0x42
	.value	0x33c
	.long	0x30
	.long	0x94cb
	.uleb128 0xc
	.long	0x949a
	.byte	0
	.uleb128 0x68
	.long	.LASF1397
	.byte	0x42
	.value	0x33e
	.long	0x30
	.long	0x94e1
	.uleb128 0xc
	.long	0x949a
	.byte	0
	.uleb128 0x67
	.long	.LASF1398
	.byte	0x42
	.byte	0xf2
	.long	0x30
	.long	0x94f6
	.uleb128 0xc
	.long	0x949a
	.byte	0
	.uleb128 0x68
	.long	.LASF1399
	.byte	0x42
	.value	0x213
	.long	0x30
	.long	0x950c
	.uleb128 0xc
	.long	0x949a
	.byte	0
	.uleb128 0x68
	.long	.LASF1400
	.byte	0x42
	.value	0x31e
	.long	0x30
	.long	0x9527
	.uleb128 0xc
	.long	0x949a
	.uleb128 0xc
	.long	0x9527
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x947d
	.uleb128 0x67
	.long	.LASF1401
	.byte	0x8
	.byte	0xfd
	.long	0x919c
	.long	0x954c
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x949a
	.byte	0
	.uleb128 0x68
	.long	.LASF1402
	.byte	0x42
	.value	0x110
	.long	0x949a
	.long	0x9567
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x68
	.long	.LASF1403
	.byte	0x8
	.value	0x11a
	.long	0x911b
	.long	0x958c
	.uleb128 0xc
	.long	0x919a
	.uleb128 0xc
	.long	0x911b
	.uleb128 0xc
	.long	0x911b
	.uleb128 0xc
	.long	0x949a
	.byte	0
	.uleb128 0x68
	.long	.LASF1404
	.byte	0x42
	.value	0x116
	.long	0x949a
	.long	0x95ac
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x949a
	.byte	0
	.uleb128 0x68
	.long	.LASF1405
	.byte	0x42
	.value	0x2ed
	.long	0x30
	.long	0x95cc
	.uleb128 0xc
	.long	0x949a
	.uleb128 0xc
	.long	0x915b
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x68
	.long	.LASF1406
	.byte	0x42
	.value	0x323
	.long	0x30
	.long	0x95e7
	.uleb128 0xc
	.long	0x949a
	.uleb128 0xc
	.long	0x95e7
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x95ed
	.uleb128 0x14
	.long	0x947d
	.uleb128 0x68
	.long	.LASF1407
	.byte	0x42
	.value	0x2f2
	.long	0x915b
	.long	0x9608
	.uleb128 0xc
	.long	0x949a
	.byte	0
	.uleb128 0x68
	.long	.LASF1408
	.byte	0x42
	.value	0x214
	.long	0x30
	.long	0x961e
	.uleb128 0xc
	.long	0x949a
	.byte	0
	.uleb128 0x69
	.long	.LASF1416
	.byte	0x47
	.byte	0x2c
	.long	0x30
	.uleb128 0x68
	.long	.LASF1409
	.byte	0x42
	.value	0x27e
	.long	0x919c
	.long	0x963f
	.uleb128 0xc
	.long	0x919c
	.byte	0
	.uleb128 0x66
	.long	.LASF1410
	.byte	0x42
	.value	0x34e
	.long	0x9651
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x67
	.long	.LASF1411
	.byte	0x42
	.byte	0xb2
	.long	0x30
	.long	0x9666
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x67
	.long	.LASF1412
	.byte	0x42
	.byte	0xb4
	.long	0x30
	.long	0x9680
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x66
	.long	.LASF1413
	.byte	0x42
	.value	0x2f7
	.long	0x9692
	.uleb128 0xc
	.long	0x949a
	.byte	0
	.uleb128 0x66
	.long	.LASF1414
	.byte	0x42
	.value	0x14c
	.long	0x96a9
	.uleb128 0xc
	.long	0x949a
	.uleb128 0xc
	.long	0x919c
	.byte	0
	.uleb128 0x68
	.long	.LASF1415
	.byte	0x42
	.value	0x150
	.long	0x30
	.long	0x96ce
	.uleb128 0xc
	.long	0x949a
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x69
	.long	.LASF1417
	.byte	0x42
	.byte	0xc3
	.long	0x949a
	.uleb128 0x67
	.long	.LASF1418
	.byte	0x42
	.byte	0xd1
	.long	0x919c
	.long	0x96ee
	.uleb128 0xc
	.long	0x919c
	.byte	0
	.uleb128 0x68
	.long	.LASF1419
	.byte	0x42
	.value	0x2be
	.long	0x30
	.long	0x9709
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x949a
	.byte	0
	.uleb128 0x37
	.long	.LASF1420
	.byte	0x40
	.value	0x165
	.long	0x913b
	.uleb128 0x16
	.long	.LASF1421
	.byte	0x44
	.byte	0x6a
	.long	0x9390
	.uleb128 0x63
	.byte	0x8
	.long	0x9726
	.uleb128 0x14
	.long	0x30
	.uleb128 0x68
	.long	.LASF1422
	.byte	0x44
	.value	0x187
	.long	0x9709
	.long	0x9741
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x68
	.long	.LASF1423
	.byte	0x44
	.value	0x2ec
	.long	0x9709
	.long	0x9757
	.uleb128 0xc
	.long	0x9757
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x9331
	.uleb128 0x68
	.long	.LASF1424
	.byte	0x48
	.value	0x180
	.long	0x977d
	.long	0x977d
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x9757
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x9783
	.uleb128 0x2
	.byte	0x4
	.byte	0x5
	.long	.LASF1425
	.uleb128 0x68
	.long	.LASF1426
	.byte	0x44
	.value	0x2fa
	.long	0x9709
	.long	0x97a5
	.uleb128 0xc
	.long	0x9783
	.uleb128 0xc
	.long	0x9757
	.byte	0
	.uleb128 0x68
	.long	.LASF1427
	.byte	0x44
	.value	0x310
	.long	0x30
	.long	0x97c0
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9757
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x97c6
	.uleb128 0x14
	.long	0x9783
	.uleb128 0x68
	.long	.LASF1428
	.byte	0x44
	.value	0x24e
	.long	0x30
	.long	0x97e6
	.uleb128 0xc
	.long	0x9757
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x68
	.long	.LASF1429
	.byte	0x48
	.value	0x159
	.long	0x30
	.long	0x9802
	.uleb128 0xc
	.long	0x9757
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0x6a
	.byte	0
	.uleb128 0x68
	.long	.LASF1430
	.byte	0x44
	.value	0x27e
	.long	0x30
	.long	0x981e
	.uleb128 0xc
	.long	0x9757
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0x6a
	.byte	0
	.uleb128 0x68
	.long	.LASF1431
	.byte	0x44
	.value	0x2ed
	.long	0x9709
	.long	0x9834
	.uleb128 0xc
	.long	0x9757
	.byte	0
	.uleb128 0x6b
	.long	.LASF1432
	.byte	0x44
	.value	0x2f3
	.long	0x9709
	.uleb128 0x68
	.long	.LASF1433
	.byte	0x44
	.value	0x192
	.long	0x911b
	.long	0x9860
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x911b
	.uleb128 0xc
	.long	0x9860
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x9715
	.uleb128 0x68
	.long	.LASF1434
	.byte	0x44
	.value	0x170
	.long	0x911b
	.long	0x988b
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x911b
	.uleb128 0xc
	.long	0x9860
	.byte	0
	.uleb128 0x68
	.long	.LASF1435
	.byte	0x44
	.value	0x16c
	.long	0x30
	.long	0x98a1
	.uleb128 0xc
	.long	0x98a1
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x98a7
	.uleb128 0x14
	.long	0x9715
	.uleb128 0x68
	.long	.LASF1436
	.byte	0x48
	.value	0x1da
	.long	0x911b
	.long	0x98d1
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x98d1
	.uleb128 0xc
	.long	0x911b
	.uleb128 0xc
	.long	0x9860
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x9472
	.uleb128 0x68
	.long	.LASF1437
	.byte	0x44
	.value	0x2fb
	.long	0x9709
	.long	0x98f2
	.uleb128 0xc
	.long	0x9783
	.uleb128 0xc
	.long	0x9757
	.byte	0
	.uleb128 0x68
	.long	.LASF1438
	.byte	0x44
	.value	0x301
	.long	0x9709
	.long	0x9908
	.uleb128 0xc
	.long	0x9783
	.byte	0
	.uleb128 0x68
	.long	.LASF1439
	.byte	0x48
	.value	0x11d
	.long	0x30
	.long	0x9929
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x911b
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0x6a
	.byte	0
	.uleb128 0x68
	.long	.LASF1440
	.byte	0x44
	.value	0x288
	.long	0x30
	.long	0x9945
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0x6a
	.byte	0
	.uleb128 0x68
	.long	.LASF1441
	.byte	0x44
	.value	0x318
	.long	0x9709
	.long	0x9960
	.uleb128 0xc
	.long	0x9709
	.uleb128 0xc
	.long	0x9757
	.byte	0
	.uleb128 0x68
	.long	.LASF1442
	.byte	0x48
	.value	0x16c
	.long	0x30
	.long	0x9980
	.uleb128 0xc
	.long	0x9757
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9980
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x93cb
	.uleb128 0x68
	.long	.LASF1443
	.byte	0x44
	.value	0x2b4
	.long	0x30
	.long	0x99a6
	.uleb128 0xc
	.long	0x9757
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9980
	.byte	0
	.uleb128 0x68
	.long	.LASF1444
	.byte	0x48
	.value	0x13b
	.long	0x30
	.long	0x99cb
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x911b
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9980
	.byte	0
	.uleb128 0x68
	.long	.LASF1445
	.byte	0x44
	.value	0x2c0
	.long	0x30
	.long	0x99eb
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9980
	.byte	0
	.uleb128 0x68
	.long	.LASF1446
	.byte	0x48
	.value	0x166
	.long	0x30
	.long	0x9a06
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9980
	.byte	0
	.uleb128 0x68
	.long	.LASF1447
	.byte	0x44
	.value	0x2bc
	.long	0x30
	.long	0x9a21
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9980
	.byte	0
	.uleb128 0x68
	.long	.LASF1448
	.byte	0x48
	.value	0x1b8
	.long	0x911b
	.long	0x9a41
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x9783
	.uleb128 0xc
	.long	0x9860
	.byte	0
	.uleb128 0x67
	.long	.LASF1449
	.byte	0x48
	.byte	0xf6
	.long	0x977d
	.long	0x9a5b
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x97c0
	.byte	0
	.uleb128 0x67
	.long	.LASF1450
	.byte	0x44
	.byte	0xa6
	.long	0x30
	.long	0x9a75
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x97c0
	.byte	0
	.uleb128 0x67
	.long	.LASF1451
	.byte	0x44
	.byte	0xc3
	.long	0x30
	.long	0x9a8f
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x97c0
	.byte	0
	.uleb128 0x67
	.long	.LASF1452
	.byte	0x48
	.byte	0x98
	.long	0x977d
	.long	0x9aa9
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x97c0
	.byte	0
	.uleb128 0x67
	.long	.LASF1453
	.byte	0x44
	.byte	0xff
	.long	0x911b
	.long	0x9ac3
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x97c0
	.byte	0
	.uleb128 0x68
	.long	.LASF1454
	.byte	0x44
	.value	0x35a
	.long	0x911b
	.long	0x9ae8
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x911b
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9ae8
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x9b7e
	.uleb128 0x6c
	.string	"tm"
	.byte	0x38
	.byte	0x49
	.byte	0x85
	.long	0x9b7e
	.uleb128 0x9
	.long	.LASF1455
	.byte	0x49
	.byte	0x87
	.long	0x30
	.byte	0
	.uleb128 0x9
	.long	.LASF1456
	.byte	0x49
	.byte	0x88
	.long	0x30
	.byte	0x4
	.uleb128 0x9
	.long	.LASF1457
	.byte	0x49
	.byte	0x89
	.long	0x30
	.byte	0x8
	.uleb128 0x9
	.long	.LASF1458
	.byte	0x49
	.byte	0x8a
	.long	0x30
	.byte	0xc
	.uleb128 0x9
	.long	.LASF1459
	.byte	0x49
	.byte	0x8b
	.long	0x30
	.byte	0x10
	.uleb128 0x9
	.long	.LASF1460
	.byte	0x49
	.byte	0x8c
	.long	0x30
	.byte	0x14
	.uleb128 0x9
	.long	.LASF1461
	.byte	0x49
	.byte	0x8d
	.long	0x30
	.byte	0x18
	.uleb128 0x9
	.long	.LASF1462
	.byte	0x49
	.byte	0x8e
	.long	0x30
	.byte	0x1c
	.uleb128 0x9
	.long	.LASF1463
	.byte	0x49
	.byte	0x8f
	.long	0x30
	.byte	0x20
	.uleb128 0x9
	.long	.LASF1464
	.byte	0x49
	.byte	0x92
	.long	0x915b
	.byte	0x28
	.uleb128 0x9
	.long	.LASF1465
	.byte	0x49
	.byte	0x93
	.long	0x9472
	.byte	0x30
	.byte	0
	.uleb128 0x14
	.long	0x9aee
	.uleb128 0x68
	.long	.LASF1466
	.byte	0x44
	.value	0x122
	.long	0x911b
	.long	0x9b99
	.uleb128 0xc
	.long	0x97c0
	.byte	0
	.uleb128 0x68
	.long	.LASF1467
	.byte	0x48
	.value	0x107
	.long	0x977d
	.long	0x9bb9
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x67
	.long	.LASF1468
	.byte	0x44
	.byte	0xa9
	.long	0x30
	.long	0x9bd8
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x67
	.long	.LASF1469
	.byte	0x48
	.byte	0xbf
	.long	0x977d
	.long	0x9bf7
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x68
	.long	.LASF1470
	.byte	0x48
	.value	0x1fc
	.long	0x911b
	.long	0x9c1c
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x9c1c
	.uleb128 0xc
	.long	0x911b
	.uleb128 0xc
	.long	0x9860
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x97c0
	.uleb128 0x68
	.long	.LASF1471
	.byte	0x44
	.value	0x103
	.long	0x911b
	.long	0x9c3d
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x97c0
	.byte	0
	.uleb128 0x68
	.long	.LASF1472
	.byte	0x44
	.value	0x1c5
	.long	0x29
	.long	0x9c58
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9c58
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x977d
	.uleb128 0x68
	.long	.LASF1473
	.byte	0x44
	.value	0x1cc
	.long	0x9c79
	.long	0x9c79
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9c58
	.byte	0
	.uleb128 0x2
	.byte	0x4
	.byte	0x4
	.long	.LASF1474
	.uleb128 0x68
	.long	.LASF1475
	.byte	0x44
	.value	0x11d
	.long	0x977d
	.long	0x9ca0
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9c58
	.byte	0
	.uleb128 0x68
	.long	.LASF1476
	.byte	0x44
	.value	0x1d7
	.long	0x915b
	.long	0x9cc0
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9c58
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x68
	.long	.LASF1477
	.byte	0x44
	.value	0x1dc
	.long	0x9126
	.long	0x9ce0
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9c58
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x67
	.long	.LASF1478
	.byte	0x44
	.byte	0xc7
	.long	0x911b
	.long	0x9cff
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x68
	.long	.LASF1479
	.byte	0x44
	.value	0x18d
	.long	0x30
	.long	0x9d15
	.uleb128 0xc
	.long	0x9709
	.byte	0
	.uleb128 0x68
	.long	.LASF1480
	.byte	0x44
	.value	0x148
	.long	0x30
	.long	0x9d35
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x67
	.long	.LASF1481
	.byte	0x48
	.byte	0x27
	.long	0x977d
	.long	0x9d54
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x67
	.long	.LASF1482
	.byte	0x48
	.byte	0x44
	.long	0x977d
	.long	0x9d73
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x67
	.long	.LASF1483
	.byte	0x48
	.byte	0x81
	.long	0x977d
	.long	0x9d92
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x9783
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x68
	.long	.LASF1484
	.byte	0x48
	.value	0x153
	.long	0x30
	.long	0x9da9
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0x6a
	.byte	0
	.uleb128 0x68
	.long	.LASF1485
	.byte	0x44
	.value	0x285
	.long	0x30
	.long	0x9dc0
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0x6a
	.byte	0
	.uleb128 0x57
	.long	.LASF1486
	.byte	0x44
	.byte	0xe3
	.long	.LASF1486
	.long	0x97c0
	.long	0x9dde
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9783
	.byte	0
	.uleb128 0x1b
	.long	.LASF1487
	.byte	0x44
	.value	0x109
	.long	.LASF1487
	.long	0x97c0
	.long	0x9dfd
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x97c0
	.byte	0
	.uleb128 0x57
	.long	.LASF1488
	.byte	0x44
	.byte	0xed
	.long	.LASF1488
	.long	0x97c0
	.long	0x9e1b
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9783
	.byte	0
	.uleb128 0x1b
	.long	.LASF1489
	.byte	0x44
	.value	0x114
	.long	.LASF1489
	.long	0x97c0
	.long	0x9e3a
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x97c0
	.byte	0
	.uleb128 0x1b
	.long	.LASF1490
	.byte	0x44
	.value	0x13f
	.long	.LASF1490
	.long	0x97c0
	.long	0x9e5e
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9783
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x68
	.long	.LASF1491
	.byte	0x44
	.value	0x1ce
	.long	0x9e79
	.long	0x9e79
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9c58
	.byte	0
	.uleb128 0x2
	.byte	0x10
	.byte	0x4
	.long	.LASF1492
	.uleb128 0x68
	.long	.LASF1493
	.byte	0x44
	.value	0x1e6
	.long	0x9ea0
	.long	0x9ea0
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9c58
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x2
	.byte	0x8
	.byte	0x5
	.long	.LASF1494
	.uleb128 0x68
	.long	.LASF1495
	.byte	0x44
	.value	0x1ed
	.long	0x9ec7
	.long	0x9ec7
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x9c58
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x2
	.byte	0x8
	.byte	0x7
	.long	.LASF1496
	.uleb128 0x63
	.byte	0x8
	.long	0x1cf2
	.uleb128 0x63
	.byte	0x8
	.long	0x1eab
	.uleb128 0x6d
	.byte	0x8
	.long	0x1eab
	.uleb128 0x6e
	.long	.LASF3074
	.uleb128 0x6f
	.byte	0x8
	.long	0x1cf2
	.uleb128 0x6d
	.byte	0x8
	.long	0x1cf2
	.uleb128 0x2
	.byte	0x1
	.byte	0x2
	.long	.LASF1497
	.uleb128 0x63
	.byte	0x8
	.long	0x1ec8
	.uleb128 0x14
	.long	0x9ef1
	.uleb128 0x63
	.byte	0x8
	.long	0x1f51
	.uleb128 0x63
	.byte	0x8
	.long	0x1fa9
	.uleb128 0x14
	.long	0x9126
	.uleb128 0x5
	.long	.LASF1498
	.byte	0x2c
	.byte	0x37
	.long	0x9f27
	.uleb128 0x22
	.byte	0x2c
	.byte	0x38
	.long	0x1ff7
	.byte	0
	.uleb128 0x6d
	.byte	0x8
	.long	0x200a
	.uleb128 0x6d
	.byte	0x8
	.long	0x203a
	.uleb128 0x63
	.byte	0x8
	.long	0x203a
	.uleb128 0x63
	.byte	0x8
	.long	0x200a
	.uleb128 0x6d
	.byte	0x8
	.long	0x2161
	.uleb128 0x16
	.long	.LASF1499
	.byte	0x4a
	.byte	0x24
	.long	0x9142
	.uleb128 0x16
	.long	.LASF1500
	.byte	0x4a
	.byte	0x25
	.long	0x9149
	.uleb128 0x16
	.long	.LASF1501
	.byte	0x4a
	.byte	0x26
	.long	0x30
	.uleb128 0x16
	.long	.LASF1502
	.byte	0x4a
	.byte	0x28
	.long	0x915b
	.uleb128 0x16
	.long	.LASF1503
	.byte	0x4a
	.byte	0x30
	.long	0x912d
	.uleb128 0x16
	.long	.LASF1504
	.byte	0x4a
	.byte	0x31
	.long	0x9134
	.uleb128 0x16
	.long	.LASF1505
	.byte	0x4a
	.byte	0x33
	.long	0x913b
	.uleb128 0x16
	.long	.LASF1506
	.byte	0x4a
	.byte	0x37
	.long	0x9126
	.uleb128 0x16
	.long	.LASF1507
	.byte	0x4a
	.byte	0x41
	.long	0x9142
	.uleb128 0x16
	.long	.LASF1508
	.byte	0x4a
	.byte	0x42
	.long	0x9149
	.uleb128 0x16
	.long	.LASF1509
	.byte	0x4a
	.byte	0x43
	.long	0x30
	.uleb128 0x16
	.long	.LASF1510
	.byte	0x4a
	.byte	0x45
	.long	0x915b
	.uleb128 0x16
	.long	.LASF1511
	.byte	0x4a
	.byte	0x4c
	.long	0x912d
	.uleb128 0x16
	.long	.LASF1512
	.byte	0x4a
	.byte	0x4d
	.long	0x9134
	.uleb128 0x16
	.long	.LASF1513
	.byte	0x4a
	.byte	0x4e
	.long	0x913b
	.uleb128 0x16
	.long	.LASF1514
	.byte	0x4a
	.byte	0x50
	.long	0x9126
	.uleb128 0x16
	.long	.LASF1515
	.byte	0x4a
	.byte	0x5a
	.long	0x9142
	.uleb128 0x16
	.long	.LASF1516
	.byte	0x4a
	.byte	0x5c
	.long	0x915b
	.uleb128 0x16
	.long	.LASF1517
	.byte	0x4a
	.byte	0x5d
	.long	0x915b
	.uleb128 0x16
	.long	.LASF1518
	.byte	0x4a
	.byte	0x5e
	.long	0x915b
	.uleb128 0x16
	.long	.LASF1519
	.byte	0x4a
	.byte	0x67
	.long	0x912d
	.uleb128 0x16
	.long	.LASF1520
	.byte	0x4a
	.byte	0x69
	.long	0x9126
	.uleb128 0x16
	.long	.LASF1521
	.byte	0x4a
	.byte	0x6a
	.long	0x9126
	.uleb128 0x16
	.long	.LASF1522
	.byte	0x4a
	.byte	0x6b
	.long	0x9126
	.uleb128 0x16
	.long	.LASF1523
	.byte	0x4a
	.byte	0x77
	.long	0x915b
	.uleb128 0x16
	.long	.LASF1524
	.byte	0x4a
	.byte	0x7a
	.long	0x9126
	.uleb128 0x16
	.long	.LASF1525
	.byte	0x4a
	.byte	0x86
	.long	0x915b
	.uleb128 0x16
	.long	.LASF1526
	.byte	0x4a
	.byte	0x87
	.long	0x9126
	.uleb128 0x2
	.byte	0x2
	.byte	0x10
	.long	.LASF1527
	.uleb128 0x2
	.byte	0x4
	.byte	0x10
	.long	.LASF1528
	.uleb128 0x7
	.long	.LASF1529
	.byte	0x60
	.byte	0x4b
	.byte	0x35
	.long	0xa1b4
	.uleb128 0x9
	.long	.LASF1530
	.byte	0x4b
	.byte	0x39
	.long	0x919c
	.byte	0
	.uleb128 0x9
	.long	.LASF1531
	.byte	0x4b
	.byte	0x3a
	.long	0x919c
	.byte	0x8
	.uleb128 0x9
	.long	.LASF1532
	.byte	0x4b
	.byte	0x40
	.long	0x919c
	.byte	0x10
	.uleb128 0x9
	.long	.LASF1533
	.byte	0x4b
	.byte	0x46
	.long	0x919c
	.byte	0x18
	.uleb128 0x9
	.long	.LASF1534
	.byte	0x4b
	.byte	0x47
	.long	0x919c
	.byte	0x20
	.uleb128 0x9
	.long	.LASF1535
	.byte	0x4b
	.byte	0x48
	.long	0x919c
	.byte	0x28
	.uleb128 0x9
	.long	.LASF1536
	.byte	0x4b
	.byte	0x49
	.long	0x919c
	.byte	0x30
	.uleb128 0x9
	.long	.LASF1537
	.byte	0x4b
	.byte	0x4a
	.long	0x919c
	.byte	0x38
	.uleb128 0x9
	.long	.LASF1538
	.byte	0x4b
	.byte	0x4b
	.long	0x919c
	.byte	0x40
	.uleb128 0x9
	.long	.LASF1539
	.byte	0x4b
	.byte	0x4c
	.long	0x919c
	.byte	0x48
	.uleb128 0x9
	.long	.LASF1540
	.byte	0x4b
	.byte	0x4d
	.long	0x91a2
	.byte	0x50
	.uleb128 0x9
	.long	.LASF1541
	.byte	0x4b
	.byte	0x4e
	.long	0x91a2
	.byte	0x51
	.uleb128 0x9
	.long	.LASF1542
	.byte	0x4b
	.byte	0x50
	.long	0x91a2
	.byte	0x52
	.uleb128 0x9
	.long	.LASF1543
	.byte	0x4b
	.byte	0x52
	.long	0x91a2
	.byte	0x53
	.uleb128 0x9
	.long	.LASF1544
	.byte	0x4b
	.byte	0x54
	.long	0x91a2
	.byte	0x54
	.uleb128 0x9
	.long	.LASF1545
	.byte	0x4b
	.byte	0x56
	.long	0x91a2
	.byte	0x55
	.uleb128 0x9
	.long	.LASF1546
	.byte	0x4b
	.byte	0x5d
	.long	0x91a2
	.byte	0x56
	.uleb128 0x9
	.long	.LASF1547
	.byte	0x4b
	.byte	0x5e
	.long	0x91a2
	.byte	0x57
	.uleb128 0x9
	.long	.LASF1548
	.byte	0x4b
	.byte	0x61
	.long	0x91a2
	.byte	0x58
	.uleb128 0x9
	.long	.LASF1549
	.byte	0x4b
	.byte	0x63
	.long	0x91a2
	.byte	0x59
	.uleb128 0x9
	.long	.LASF1550
	.byte	0x4b
	.byte	0x65
	.long	0x91a2
	.byte	0x5a
	.uleb128 0x9
	.long	.LASF1551
	.byte	0x4b
	.byte	0x67
	.long	0x91a2
	.byte	0x5b
	.uleb128 0x9
	.long	.LASF1552
	.byte	0x4b
	.byte	0x6e
	.long	0x91a2
	.byte	0x5c
	.uleb128 0x9
	.long	.LASF1553
	.byte	0x4b
	.byte	0x6f
	.long	0x91a2
	.byte	0x5d
	.byte	0
	.uleb128 0x67
	.long	.LASF1554
	.byte	0x4b
	.byte	0x7c
	.long	0x919c
	.long	0xa1ce
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x69
	.long	.LASF1555
	.byte	0x4b
	.byte	0x7f
	.long	0xa1d9
	.uleb128 0x63
	.byte	0x8
	.long	0xa087
	.uleb128 0x16
	.long	.LASF1556
	.byte	0x49
	.byte	0x3b
	.long	0x918f
	.uleb128 0x16
	.long	.LASF1557
	.byte	0x4c
	.byte	0x20
	.long	0x30
	.uleb128 0x63
	.byte	0x8
	.long	0xa1fb
	.uleb128 0x70
	.uleb128 0x6d
	.byte	0x8
	.long	0x91a2
	.uleb128 0x6d
	.byte	0x8
	.long	0x9478
	.uleb128 0x63
	.byte	0x8
	.long	0x79aa
	.uleb128 0x6d
	.byte	0x8
	.long	0x7b04
	.uleb128 0x63
	.byte	0x8
	.long	0x7b04
	.uleb128 0x63
	.byte	0x8
	.long	0x22c8
	.uleb128 0x6d
	.byte	0x8
	.long	0x2330
	.uleb128 0x64
	.byte	0x8
	.byte	0x4d
	.byte	0x62
	.long	.LASF1558
	.long	0xa24b
	.uleb128 0x9
	.long	.LASF1559
	.byte	0x4d
	.byte	0x63
	.long	0x30
	.byte	0
	.uleb128 0x71
	.string	"rem"
	.byte	0x4d
	.byte	0x64
	.long	0x30
	.byte	0x4
	.byte	0
	.uleb128 0x16
	.long	.LASF1560
	.byte	0x4d
	.byte	0x65
	.long	0xa226
	.uleb128 0x64
	.byte	0x10
	.byte	0x4d
	.byte	0x6a
	.long	.LASF1561
	.long	0xa27b
	.uleb128 0x9
	.long	.LASF1559
	.byte	0x4d
	.byte	0x6b
	.long	0x915b
	.byte	0
	.uleb128 0x71
	.string	"rem"
	.byte	0x4d
	.byte	0x6c
	.long	0x915b
	.byte	0x8
	.byte	0
	.uleb128 0x16
	.long	.LASF1562
	.byte	0x4d
	.byte	0x6d
	.long	0xa256
	.uleb128 0x64
	.byte	0x10
	.byte	0x4d
	.byte	0x76
	.long	.LASF1563
	.long	0xa2ab
	.uleb128 0x9
	.long	.LASF1559
	.byte	0x4d
	.byte	0x77
	.long	0x9ea0
	.byte	0
	.uleb128 0x71
	.string	"rem"
	.byte	0x4d
	.byte	0x78
	.long	0x9ea0
	.byte	0x8
	.byte	0
	.uleb128 0x16
	.long	.LASF1564
	.byte	0x4d
	.byte	0x79
	.long	0xa286
	.uleb128 0x37
	.long	.LASF1565
	.byte	0x4d
	.value	0x2e5
	.long	0xa2c2
	.uleb128 0x63
	.byte	0x8
	.long	0xa2c8
	.uleb128 0x72
	.long	0x30
	.long	0xa2dc
	.uleb128 0xc
	.long	0xa1f5
	.uleb128 0xc
	.long	0xa1f5
	.byte	0
	.uleb128 0x68
	.long	.LASF1566
	.byte	0x4d
	.value	0x207
	.long	0x30
	.long	0xa2f2
	.uleb128 0xc
	.long	0xa2f2
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0xa2f8
	.uleb128 0x73
	.uleb128 0x1b
	.long	.LASF1567
	.byte	0x4d
	.value	0x20c
	.long	.LASF1567
	.long	0x30
	.long	0xa313
	.uleb128 0xc
	.long	0xa2f2
	.byte	0
	.uleb128 0x74
	.long	.LASF1568
	.byte	0x13
	.byte	0x1a
	.long	0x29
	.byte	0x3
	.long	0xa32f
	.uleb128 0x75
	.long	.LASF2775
	.byte	0x13
	.byte	0x1a
	.long	0x9472
	.byte	0
	.uleb128 0x68
	.long	.LASF1569
	.byte	0x4d
	.value	0x116
	.long	0x30
	.long	0xa345
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x68
	.long	.LASF1570
	.byte	0x4d
	.value	0x11b
	.long	0x915b
	.long	0xa35b
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x67
	.long	.LASF1571
	.byte	0x4e
	.byte	0x14
	.long	0x919a
	.long	0xa384
	.uleb128 0xc
	.long	0xa1f5
	.uleb128 0xc
	.long	0xa1f5
	.uleb128 0xc
	.long	0x911b
	.uleb128 0xc
	.long	0x911b
	.uleb128 0xc
	.long	0xa2b6
	.byte	0
	.uleb128 0x76
	.string	"div"
	.byte	0x4d
	.value	0x314
	.long	0xa24b
	.long	0xa39f
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x68
	.long	.LASF1572
	.byte	0x4d
	.value	0x234
	.long	0x919c
	.long	0xa3b5
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x68
	.long	.LASF1573
	.byte	0x4d
	.value	0x316
	.long	0xa27b
	.long	0xa3d0
	.uleb128 0xc
	.long	0x915b
	.uleb128 0xc
	.long	0x915b
	.byte	0
	.uleb128 0x68
	.long	.LASF1574
	.byte	0x4d
	.value	0x35e
	.long	0x30
	.long	0xa3eb
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x67
	.long	.LASF1575
	.byte	0x4f
	.byte	0x71
	.long	0x911b
	.long	0xa40a
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x68
	.long	.LASF1576
	.byte	0x4d
	.value	0x361
	.long	0x30
	.long	0xa42a
	.uleb128 0xc
	.long	0x977d
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x66
	.long	.LASF1577
	.byte	0x4d
	.value	0x2fc
	.long	0xa44b
	.uleb128 0xc
	.long	0x919a
	.uleb128 0xc
	.long	0x911b
	.uleb128 0xc
	.long	0x911b
	.uleb128 0xc
	.long	0xa2b6
	.byte	0
	.uleb128 0x77
	.long	.LASF1578
	.byte	0x4d
	.value	0x225
	.long	0xa45d
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x6b
	.long	.LASF1579
	.byte	0x4d
	.value	0x176
	.long	0x30
	.uleb128 0x66
	.long	.LASF1580
	.byte	0x4d
	.value	0x178
	.long	0xa47b
	.uleb128 0xc
	.long	0x913b
	.byte	0
	.uleb128 0x67
	.long	.LASF1581
	.byte	0x4d
	.byte	0xa4
	.long	0x29
	.long	0xa495
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xa495
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x919c
	.uleb128 0x67
	.long	.LASF1582
	.byte	0x4d
	.byte	0xb7
	.long	0x915b
	.long	0xa4ba
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xa495
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x67
	.long	.LASF1583
	.byte	0x4d
	.byte	0xbb
	.long	0x9126
	.long	0xa4d9
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xa495
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x68
	.long	.LASF1584
	.byte	0x4d
	.value	0x2cc
	.long	0x30
	.long	0xa4ef
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x67
	.long	.LASF1585
	.byte	0x4f
	.byte	0x90
	.long	0x911b
	.long	0xa50e
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x97c0
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x67
	.long	.LASF1586
	.byte	0x4f
	.byte	0x53
	.long	0x30
	.long	0xa528
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x9783
	.byte	0
	.uleb128 0x68
	.long	.LASF1587
	.byte	0x4d
	.value	0x31c
	.long	0xa2ab
	.long	0xa543
	.uleb128 0xc
	.long	0x9ea0
	.uleb128 0xc
	.long	0x9ea0
	.byte	0
	.uleb128 0x68
	.long	.LASF1588
	.byte	0x4d
	.value	0x124
	.long	0x9ea0
	.long	0xa559
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x67
	.long	.LASF1589
	.byte	0x4d
	.byte	0xd1
	.long	0x9ea0
	.long	0xa578
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xa495
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x67
	.long	.LASF1590
	.byte	0x4d
	.byte	0xd6
	.long	0x9ec7
	.long	0xa597
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xa495
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x67
	.long	.LASF1591
	.byte	0x4d
	.byte	0xac
	.long	0x9c79
	.long	0xa5b1
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xa495
	.byte	0
	.uleb128 0x67
	.long	.LASF1592
	.byte	0x4d
	.byte	0xaf
	.long	0x9e79
	.long	0xa5cb
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xa495
	.byte	0
	.uleb128 0x6d
	.byte	0x8
	.long	0x2429
	.uleb128 0x6d
	.byte	0x8
	.long	0x24ee
	.uleb128 0x6d
	.byte	0x8
	.long	0x7bc1
	.uleb128 0x6d
	.byte	0x8
	.long	0x7c03
	.uleb128 0x6d
	.byte	0x8
	.long	0x22c8
	.uleb128 0x63
	.byte	0x8
	.long	0x59
	.uleb128 0x60
	.long	0x91a2
	.long	0xa5ff
	.uleb128 0x61
	.long	0x9188
	.byte	0xf
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x4d
	.uleb128 0x63
	.byte	0x8
	.long	0x1a20
	.uleb128 0x6d
	.byte	0x8
	.long	0xf1
	.uleb128 0x6d
	.byte	0x8
	.long	0x137
	.uleb128 0x6d
	.byte	0x8
	.long	0x36c
	.uleb128 0x6d
	.byte	0x8
	.long	0x1a20
	.uleb128 0x6f
	.byte	0x8
	.long	0x4d
	.uleb128 0x6d
	.byte	0x8
	.long	0x4d
	.uleb128 0x63
	.byte	0x8
	.long	0x2538
	.uleb128 0x63
	.byte	0x8
	.long	0x2620
	.uleb128 0x6d
	.byte	0x8
	.long	0x1a30
	.uleb128 0x14
	.long	0x9472
	.uleb128 0x63
	.byte	0x8
	.long	0x279f
	.uleb128 0x16
	.long	.LASF1593
	.byte	0x50
	.byte	0x34
	.long	0x9126
	.uleb128 0x16
	.long	.LASF1594
	.byte	0x50
	.byte	0xba
	.long	0xa662
	.uleb128 0x63
	.byte	0x8
	.long	0xa668
	.uleb128 0x14
	.long	0x9150
	.uleb128 0x67
	.long	.LASF1595
	.byte	0x50
	.byte	0xaf
	.long	0x30
	.long	0xa687
	.uleb128 0xc
	.long	0x9709
	.uleb128 0xc
	.long	0xa64c
	.byte	0
	.uleb128 0x67
	.long	.LASF1596
	.byte	0x50
	.byte	0xdd
	.long	0x9709
	.long	0xa6a1
	.uleb128 0xc
	.long	0x9709
	.uleb128 0xc
	.long	0xa657
	.byte	0
	.uleb128 0x67
	.long	.LASF1597
	.byte	0x50
	.byte	0xda
	.long	0xa657
	.long	0xa6b6
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x67
	.long	.LASF1598
	.byte	0x50
	.byte	0xab
	.long	0xa64c
	.long	0xa6cb
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x6d
	.byte	0x8
	.long	0x2a30
	.uleb128 0x14
	.long	0x9149
	.uleb128 0x14
	.long	0x915b
	.uleb128 0x57
	.long	.LASF1599
	.byte	0x51
	.byte	0x55
	.long	.LASF1599
	.long	0xa1f5
	.long	0xa6fe
	.uleb128 0xc
	.long	0xa1f5
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x67
	.long	.LASF1600
	.byte	0x51
	.byte	0x93
	.long	0x30
	.long	0xa718
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x68
	.long	.LASF1601
	.byte	0x51
	.value	0x198
	.long	0x919c
	.long	0xa72e
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x68
	.long	.LASF1602
	.byte	0x51
	.value	0x157
	.long	0x919c
	.long	0xa749
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x67
	.long	.LASF1603
	.byte	0x51
	.byte	0x96
	.long	0x911b
	.long	0xa768
	.uleb128 0xc
	.long	0x919c
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x57
	.long	.LASF1604
	.byte	0x51
	.byte	0xe0
	.long	.LASF1604
	.long	0x9472
	.long	0xa786
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1b
	.long	.LASF1605
	.byte	0x51
	.value	0x12f
	.long	.LASF1605
	.long	0x9472
	.long	0xa7a5
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x57
	.long	.LASF1606
	.byte	0x51
	.byte	0xfb
	.long	.LASF1606
	.long	0x9472
	.long	0xa7c3
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1b
	.long	.LASF1607
	.byte	0x51
	.value	0x14a
	.long	.LASF1607
	.long	0x9472
	.long	0xa7e2
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x16
	.long	.LASF1608
	.byte	0x52
	.byte	0x1c
	.long	0x9c79
	.uleb128 0x16
	.long	.LASF1609
	.byte	0x52
	.byte	0x1d
	.long	0x29
	.uleb128 0x16
	.long	.LASF1610
	.byte	0x4
	.byte	0x45
	.long	0xa803
	.uleb128 0x78
	.long	0x9c79
	.long	0xa80f
	.uleb128 0x79
	.byte	0x3
	.byte	0
	.uleb128 0x16
	.long	.LASF1611
	.byte	0x4
	.byte	0x48
	.long	0xa81a
	.uleb128 0x78
	.long	0x9c79
	.long	0xa826
	.uleb128 0x79
	.byte	0x3
	.byte	0
	.uleb128 0x16
	.long	.LASF1612
	.byte	0x3
	.byte	0x28
	.long	0xa831
	.uleb128 0x78
	.long	0x29
	.long	0xa83d
	.uleb128 0x79
	.byte	0x1
	.byte	0
	.uleb128 0x16
	.long	.LASF1613
	.byte	0x3
	.byte	0x2b
	.long	0xa848
	.uleb128 0x78
	.long	0x30
	.long	0xa854
	.uleb128 0x79
	.byte	0x3
	.byte	0
	.uleb128 0x16
	.long	.LASF1614
	.byte	0x3
	.byte	0x2d
	.long	0xa85f
	.uleb128 0x78
	.long	0x9149
	.long	0xa86b
	.uleb128 0x79
	.byte	0x7
	.byte	0
	.uleb128 0x16
	.long	.LASF1615
	.byte	0x3
	.byte	0x2e
	.long	0xa876
	.uleb128 0x78
	.long	0x9134
	.long	0xa882
	.uleb128 0x79
	.byte	0x7
	.byte	0
	.uleb128 0x16
	.long	.LASF1616
	.byte	0x3
	.byte	0x2f
	.long	0xa88d
	.uleb128 0x78
	.long	0x91a2
	.long	0xa899
	.uleb128 0x79
	.byte	0xf
	.byte	0
	.uleb128 0x16
	.long	.LASF1617
	.byte	0x3
	.byte	0x34
	.long	0xa8a4
	.uleb128 0x78
	.long	0x9ea0
	.long	0xa8b0
	.uleb128 0x79
	.byte	0x1
	.byte	0
	.uleb128 0x16
	.long	.LASF1618
	.byte	0x3
	.byte	0x35
	.long	0xa8bb
	.uleb128 0x78
	.long	0x29
	.long	0xa8c7
	.uleb128 0x79
	.byte	0x1
	.byte	0
	.uleb128 0x16
	.long	.LASF1619
	.byte	0x53
	.byte	0xaa
	.long	0x912d
	.uleb128 0x16
	.long	.LASF1620
	.byte	0x53
	.byte	0xae
	.long	0x9142
	.uleb128 0x36
	.long	.LASF1621
	.byte	0x90
	.byte	0x53
	.value	0x1ec
	.long	0xaa08
	.uleb128 0x4b
	.long	.LASF1622
	.byte	0x53
	.value	0x1ee
	.long	0x30
	.byte	0
	.uleb128 0x7a
	.string	"ID"
	.byte	0x53
	.value	0x1ef
	.long	0x30
	.byte	0x4
	.uleb128 0x4b
	.long	.LASF1623
	.byte	0x53
	.value	0x1f0
	.long	0x30
	.byte	0x8
	.uleb128 0x4b
	.long	.LASF1624
	.byte	0x53
	.value	0x1f1
	.long	0x30
	.byte	0xc
	.uleb128 0x4b
	.long	.LASF1625
	.byte	0x53
	.value	0x1f2
	.long	0x30
	.byte	0x10
	.uleb128 0x4b
	.long	.LASF1626
	.byte	0x53
	.value	0x1f4
	.long	0x9380
	.byte	0x14
	.uleb128 0x4b
	.long	.LASF1627
	.byte	0x53
	.value	0x1f5
	.long	0x9380
	.byte	0x18
	.uleb128 0x4b
	.long	.LASF1628
	.byte	0x53
	.value	0x1f6
	.long	0x30
	.byte	0x1c
	.uleb128 0x4b
	.long	.LASF1629
	.byte	0x53
	.value	0x1f8
	.long	0x30
	.byte	0x20
	.uleb128 0x4b
	.long	.LASF1630
	.byte	0x53
	.value	0x1fa
	.long	0x30
	.byte	0x24
	.uleb128 0x4b
	.long	.LASF1631
	.byte	0x53
	.value	0x1fc
	.long	0x30
	.byte	0x28
	.uleb128 0x4b
	.long	.LASF1632
	.byte	0x53
	.value	0x1fd
	.long	0x30
	.byte	0x2c
	.uleb128 0x7a
	.string	"roi"
	.byte	0x53
	.value	0x1fe
	.long	0xaa57
	.byte	0x30
	.uleb128 0x4b
	.long	.LASF1633
	.byte	0x53
	.value	0x1ff
	.long	0xaa5d
	.byte	0x38
	.uleb128 0x4b
	.long	.LASF1634
	.byte	0x53
	.value	0x200
	.long	0x919a
	.byte	0x40
	.uleb128 0x4b
	.long	.LASF1635
	.byte	0x53
	.value	0x201
	.long	0xaa68
	.byte	0x48
	.uleb128 0x4b
	.long	.LASF1636
	.byte	0x53
	.value	0x202
	.long	0x30
	.byte	0x50
	.uleb128 0x4b
	.long	.LASF1637
	.byte	0x53
	.value	0x205
	.long	0x919c
	.byte	0x58
	.uleb128 0x4b
	.long	.LASF1638
	.byte	0x53
	.value	0x206
	.long	0x30
	.byte	0x60
	.uleb128 0x4b
	.long	.LASF1639
	.byte	0x53
	.value	0x207
	.long	0xaa6e
	.byte	0x64
	.uleb128 0x4b
	.long	.LASF1640
	.byte	0x53
	.value	0x208
	.long	0xaa6e
	.byte	0x74
	.uleb128 0x4b
	.long	.LASF1641
	.byte	0x53
	.value	0x209
	.long	0x919c
	.byte	0x88
	.byte	0
	.uleb128 0x36
	.long	.LASF1642
	.byte	0x14
	.byte	0x53
	.value	0x211
	.long	0xaa57
	.uleb128 0x7a
	.string	"coi"
	.byte	0x53
	.value	0x213
	.long	0x30
	.byte	0
	.uleb128 0x4b
	.long	.LASF1643
	.byte	0x53
	.value	0x214
	.long	0x30
	.byte	0x4
	.uleb128 0x4b
	.long	.LASF1644
	.byte	0x53
	.value	0x215
	.long	0x30
	.byte	0x8
	.uleb128 0x4b
	.long	.LASF1631
	.byte	0x53
	.value	0x216
	.long	0x30
	.byte	0xc
	.uleb128 0x4b
	.long	.LASF1632
	.byte	0x53
	.value	0x217
	.long	0x30
	.byte	0x10
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0xaa08
	.uleb128 0x63
	.byte	0x8
	.long	0xa8dd
	.uleb128 0x7b
	.long	.LASF1695
	.uleb128 0x63
	.byte	0x8
	.long	0xaa63
	.uleb128 0x60
	.long	0x30
	.long	0xaa7e
	.uleb128 0x61
	.long	0x9188
	.byte	0x3
	.byte	0
	.uleb128 0x37
	.long	.LASF1645
	.byte	0x53
	.value	0x20d
	.long	0xa8dd
	.uleb128 0x63
	.byte	0x8
	.long	0x30
	.uleb128 0x63
	.byte	0x8
	.long	0x9c79
	.uleb128 0x36
	.long	.LASF1646
	.byte	0x28
	.byte	0x53
	.value	0x2a0
	.long	0xab75
	.uleb128 0x7c
	.byte	0x8
	.byte	0x53
	.value	0x2aa
	.long	0xaae3
	.uleb128 0x7d
	.string	"ptr"
	.byte	0x53
	.value	0x2ab
	.long	0xab75
	.uleb128 0x7d
	.string	"s"
	.byte	0x53
	.value	0x2ac
	.long	0xab7b
	.uleb128 0x7d
	.string	"i"
	.byte	0x53
	.value	0x2ad
	.long	0xaa8a
	.uleb128 0x7d
	.string	"fl"
	.byte	0x53
	.value	0x2ae
	.long	0xaa90
	.uleb128 0x7d
	.string	"db"
	.byte	0x53
	.value	0x2af
	.long	0xab81
	.byte	0
	.uleb128 0x7c
	.byte	0x4
	.byte	0x53
	.value	0x2b4
	.long	0xab05
	.uleb128 0x7e
	.long	.LASF1647
	.byte	0x53
	.value	0x2b5
	.long	0x30
	.uleb128 0x7e
	.long	.LASF1632
	.byte	0x53
	.value	0x2b6
	.long	0x30
	.byte	0
	.uleb128 0x7c
	.byte	0x4
	.byte	0x53
	.value	0x2ba
	.long	0xab27
	.uleb128 0x7e
	.long	.LASF1648
	.byte	0x53
	.value	0x2bb
	.long	0x30
	.uleb128 0x7e
	.long	.LASF1631
	.byte	0x53
	.value	0x2bc
	.long	0x30
	.byte	0
	.uleb128 0x4b
	.long	.LASF1649
	.byte	0x53
	.value	0x2a2
	.long	0x30
	.byte	0
	.uleb128 0x4b
	.long	.LASF1650
	.byte	0x53
	.value	0x2a3
	.long	0x30
	.byte	0x4
	.uleb128 0x4b
	.long	.LASF1651
	.byte	0x53
	.value	0x2a6
	.long	0xaa8a
	.byte	0x8
	.uleb128 0x4b
	.long	.LASF1652
	.byte	0x53
	.value	0x2a7
	.long	0x30
	.byte	0x10
	.uleb128 0x4b
	.long	.LASF209
	.byte	0x53
	.value	0x2b0
	.long	0xaaa3
	.byte	0x18
	.uleb128 0x15
	.long	0xaae3
	.byte	0x20
	.uleb128 0x15
	.long	0xab05
	.byte	0x24
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0xa8c7
	.uleb128 0x63
	.byte	0x8
	.long	0x9149
	.uleb128 0x63
	.byte	0x8
	.long	0x29
	.uleb128 0x37
	.long	.LASF1646
	.byte	0x53
	.value	0x2c4
	.long	0xaa96
	.uleb128 0x7f
	.long	.LASF1653
	.value	0x120
	.byte	0x53
	.value	0x34b
	.long	0xac55
	.uleb128 0x7c
	.byte	0x8
	.byte	0x53
	.value	0x354
	.long	0xabe1
	.uleb128 0x7d
	.string	"ptr"
	.byte	0x53
	.value	0x355
	.long	0xab75
	.uleb128 0x7d
	.string	"fl"
	.byte	0x53
	.value	0x356
	.long	0xaa90
	.uleb128 0x7d
	.string	"db"
	.byte	0x53
	.value	0x357
	.long	0xab81
	.uleb128 0x7d
	.string	"i"
	.byte	0x53
	.value	0x358
	.long	0xaa8a
	.uleb128 0x7d
	.string	"s"
	.byte	0x53
	.value	0x359
	.long	0xab7b
	.byte	0
	.uleb128 0x80
	.byte	0x8
	.byte	0x53
	.value	0x35d
	.long	0xac06
	.uleb128 0x4b
	.long	.LASF115
	.byte	0x53
	.value	0x35e
	.long	0x30
	.byte	0
	.uleb128 0x4b
	.long	.LASF1650
	.byte	0x53
	.value	0x35f
	.long	0x30
	.byte	0x4
	.byte	0
	.uleb128 0x4b
	.long	.LASF1649
	.byte	0x53
	.value	0x34d
	.long	0x30
	.byte	0
	.uleb128 0x4b
	.long	.LASF1654
	.byte	0x53
	.value	0x34e
	.long	0x30
	.byte	0x4
	.uleb128 0x4b
	.long	.LASF1651
	.byte	0x53
	.value	0x350
	.long	0xaa8a
	.byte	0x8
	.uleb128 0x4b
	.long	.LASF1652
	.byte	0x53
	.value	0x351
	.long	0x30
	.byte	0x10
	.uleb128 0x4b
	.long	.LASF209
	.byte	0x53
	.value	0x35a
	.long	0xaba1
	.byte	0x18
	.uleb128 0x7a
	.string	"dim"
	.byte	0x53
	.value	0x361
	.long	0xac55
	.byte	0x20
	.byte	0
	.uleb128 0x60
	.long	0xabe1
	.long	0xac65
	.uleb128 0x61
	.long	0x9188
	.byte	0x1f
	.byte	0
	.uleb128 0x37
	.long	.LASF1653
	.byte	0x53
	.value	0x363
	.long	0xab93
	.uleb128 0x63
	.byte	0x8
	.long	0x919a
	.uleb128 0x36
	.long	.LASF1655
	.byte	0x10
	.byte	0x53
	.value	0x3d2
	.long	0xacb5
	.uleb128 0x7a
	.string	"x"
	.byte	0x53
	.value	0x3d4
	.long	0x30
	.byte	0
	.uleb128 0x7a
	.string	"y"
	.byte	0x53
	.value	0x3d5
	.long	0x30
	.byte	0x4
	.uleb128 0x4b
	.long	.LASF1631
	.byte	0x53
	.value	0x3d6
	.long	0x30
	.byte	0x8
	.uleb128 0x4b
	.long	.LASF1632
	.byte	0x53
	.value	0x3d7
	.long	0x30
	.byte	0xc
	.byte	0
	.uleb128 0x37
	.long	.LASF1655
	.byte	0x53
	.value	0x3d9
	.long	0xac77
	.uleb128 0x36
	.long	.LASF1656
	.byte	0x8
	.byte	0x53
	.value	0x418
	.long	0xace5
	.uleb128 0x7a
	.string	"x"
	.byte	0x53
	.value	0x41a
	.long	0x30
	.byte	0
	.uleb128 0x7a
	.string	"y"
	.byte	0x53
	.value	0x41b
	.long	0x30
	.byte	0x4
	.byte	0
	.uleb128 0x37
	.long	.LASF1656
	.byte	0x53
	.value	0x41d
	.long	0xacc1
	.uleb128 0x36
	.long	.LASF1657
	.byte	0x8
	.byte	0x53
	.value	0x42b
	.long	0xad15
	.uleb128 0x7a
	.string	"x"
	.byte	0x53
	.value	0x42d
	.long	0x9c79
	.byte	0
	.uleb128 0x7a
	.string	"y"
	.byte	0x53
	.value	0x42e
	.long	0x9c79
	.byte	0x4
	.byte	0
	.uleb128 0x37
	.long	.LASF1657
	.byte	0x53
	.value	0x430
	.long	0xacf1
	.uleb128 0x36
	.long	.LASF1658
	.byte	0x8
	.byte	0x53
	.value	0x48d
	.long	0xad49
	.uleb128 0x4b
	.long	.LASF1631
	.byte	0x53
	.value	0x48f
	.long	0x30
	.byte	0
	.uleb128 0x4b
	.long	.LASF1632
	.byte	0x53
	.value	0x490
	.long	0x30
	.byte	0x4
	.byte	0
	.uleb128 0x37
	.long	.LASF1658
	.byte	0x53
	.value	0x492
	.long	0xad21
	.uleb128 0x36
	.long	.LASF1659
	.byte	0x8
	.byte	0x53
	.value	0x49e
	.long	0xad7d
	.uleb128 0x4b
	.long	.LASF1631
	.byte	0x53
	.value	0x4a0
	.long	0x9c79
	.byte	0
	.uleb128 0x4b
	.long	.LASF1632
	.byte	0x53
	.value	0x4a1
	.long	0x9c79
	.byte	0x4
	.byte	0
	.uleb128 0x37
	.long	.LASF1659
	.byte	0x53
	.value	0x4a3
	.long	0xad55
	.uleb128 0x36
	.long	.LASF1660
	.byte	0x8
	.byte	0x53
	.value	0x4cd
	.long	0xadb1
	.uleb128 0x4b
	.long	.LASF1661
	.byte	0x53
	.value	0x4cf
	.long	0x30
	.byte	0
	.uleb128 0x4b
	.long	.LASF1662
	.byte	0x53
	.value	0x4cf
	.long	0x30
	.byte	0x4
	.byte	0
	.uleb128 0x37
	.long	.LASF1660
	.byte	0x53
	.value	0x4d1
	.long	0xad89
	.uleb128 0x36
	.long	.LASF1663
	.byte	0x20
	.byte	0x53
	.value	0x4e2
	.long	0xadd8
	.uleb128 0x7a
	.string	"val"
	.byte	0x53
	.value	0x4e4
	.long	0xadd8
	.byte	0
	.byte	0
	.uleb128 0x60
	.long	0x29
	.long	0xade8
	.uleb128 0x61
	.long	0x9188
	.byte	0x3
	.byte	0
	.uleb128 0x37
	.long	.LASF1663
	.byte	0x53
	.value	0x4e6
	.long	0xadbd
	.uleb128 0x36
	.long	.LASF1664
	.byte	0x10
	.byte	0x53
	.value	0x50a
	.long	0xae1c
	.uleb128 0x4b
	.long	.LASF1665
	.byte	0x53
	.value	0x50c
	.long	0xae1c
	.byte	0
	.uleb128 0x4b
	.long	.LASF1666
	.byte	0x53
	.value	0x50d
	.long	0xae1c
	.byte	0x8
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0xadf4
	.uleb128 0x37
	.long	.LASF1664
	.byte	0x53
	.value	0x50f
	.long	0xadf4
	.uleb128 0x36
	.long	.LASF1667
	.byte	0x28
	.byte	0x53
	.value	0x513
	.long	0xae8a
	.uleb128 0x4b
	.long	.LASF1668
	.byte	0x53
	.value	0x515
	.long	0x30
	.byte	0
	.uleb128 0x4b
	.long	.LASF1669
	.byte	0x53
	.value	0x516
	.long	0xae8a
	.byte	0x8
	.uleb128 0x7a
	.string	"top"
	.byte	0x53
	.value	0x517
	.long	0xae8a
	.byte	0x10
	.uleb128 0x4b
	.long	.LASF1670
	.byte	0x53
	.value	0x518
	.long	0xae90
	.byte	0x18
	.uleb128 0x4b
	.long	.LASF1671
	.byte	0x53
	.value	0x519
	.long	0x30
	.byte	0x20
	.uleb128 0x4b
	.long	.LASF1672
	.byte	0x53
	.value	0x51a
	.long	0x30
	.byte	0x24
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0xae22
	.uleb128 0x63
	.byte	0x8
	.long	0xae2e
	.uleb128 0x37
	.long	.LASF1667
	.byte	0x53
	.value	0x51c
	.long	0xae2e
	.uleb128 0x36
	.long	.LASF1673
	.byte	0x20
	.byte	0x53
	.value	0x52d
	.long	0xaef1
	.uleb128 0x4b
	.long	.LASF1665
	.byte	0x53
	.value	0x52f
	.long	0xaef1
	.byte	0
	.uleb128 0x4b
	.long	.LASF1666
	.byte	0x53
	.value	0x530
	.long	0xaef1
	.byte	0x8
	.uleb128 0x4b
	.long	.LASF1661
	.byte	0x53
	.value	0x531
	.long	0x30
	.byte	0x10
	.uleb128 0x4b
	.long	.LASF1674
	.byte	0x53
	.value	0x533
	.long	0x30
	.byte	0x14
	.uleb128 0x4b
	.long	.LASF209
	.byte	0x53
	.value	0x534
	.long	0xaef7
	.byte	0x18
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0xaea2
	.uleb128 0x63
	.byte	0x8
	.long	0xa8d2
	.uleb128 0x37
	.long	.LASF1673
	.byte	0x53
	.value	0x536
	.long	0xaea2
	.uleb128 0x36
	.long	.LASF1675
	.byte	0x60
	.byte	0x53
	.value	0x550
	.long	0xafcd
	.uleb128 0x4b
	.long	.LASF1676
	.byte	0x53
	.value	0x552
	.long	0x30
	.byte	0
	.uleb128 0x4b
	.long	.LASF1677
	.byte	0x53
	.value	0x552
	.long	0x30
	.byte	0x4
	.uleb128 0x4b
	.long	.LASF1678
	.byte	0x53
	.value	0x552
	.long	0xafcd
	.byte	0x8
	.uleb128 0x4b
	.long	.LASF1679
	.byte	0x53
	.value	0x552
	.long	0xafcd
	.byte	0x10
	.uleb128 0x4b
	.long	.LASF1680
	.byte	0x53
	.value	0x552
	.long	0xafcd
	.byte	0x18
	.uleb128 0x4b
	.long	.LASF1681
	.byte	0x53
	.value	0x552
	.long	0xafcd
	.byte	0x20
	.uleb128 0x4b
	.long	.LASF1682
	.byte	0x53
	.value	0x552
	.long	0x30
	.byte	0x28
	.uleb128 0x4b
	.long	.LASF1683
	.byte	0x53
	.value	0x552
	.long	0x30
	.byte	0x2c
	.uleb128 0x4b
	.long	.LASF1684
	.byte	0x53
	.value	0x552
	.long	0xaef7
	.byte	0x30
	.uleb128 0x7a
	.string	"ptr"
	.byte	0x53
	.value	0x552
	.long	0xaef7
	.byte	0x38
	.uleb128 0x4b
	.long	.LASF1685
	.byte	0x53
	.value	0x552
	.long	0x30
	.byte	0x40
	.uleb128 0x4b
	.long	.LASF1686
	.byte	0x53
	.value	0x552
	.long	0xafd3
	.byte	0x48
	.uleb128 0x4b
	.long	.LASF1687
	.byte	0x53
	.value	0x552
	.long	0xafd9
	.byte	0x50
	.uleb128 0x4b
	.long	.LASF1688
	.byte	0x53
	.value	0x552
	.long	0xafd9
	.byte	0x58
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0xaf09
	.uleb128 0x63
	.byte	0x8
	.long	0xae96
	.uleb128 0x63
	.byte	0x8
	.long	0xaefd
	.uleb128 0x37
	.long	.LASF1675
	.byte	0x53
	.value	0x554
	.long	0xaf09
	.uleb128 0x63
	.byte	0x8
	.long	0xafdf
	.uleb128 0x36
	.long	.LASF1689
	.byte	0x40
	.byte	0x53
	.value	0x66f
	.long	0xb067
	.uleb128 0x4b
	.long	.LASF1677
	.byte	0x53
	.value	0x671
	.long	0x30
	.byte	0
	.uleb128 0x7a
	.string	"seq"
	.byte	0x53
	.value	0x671
	.long	0xafeb
	.byte	0x8
	.uleb128 0x4b
	.long	.LASF1690
	.byte	0x53
	.value	0x671
	.long	0xafd9
	.byte	0x10
	.uleb128 0x7a
	.string	"ptr"
	.byte	0x53
	.value	0x671
	.long	0xaef7
	.byte	0x18
	.uleb128 0x4b
	.long	.LASF1691
	.byte	0x53
	.value	0x671
	.long	0xaef7
	.byte	0x20
	.uleb128 0x4b
	.long	.LASF1684
	.byte	0x53
	.value	0x671
	.long	0xaef7
	.byte	0x28
	.uleb128 0x4b
	.long	.LASF1692
	.byte	0x53
	.value	0x671
	.long	0x30
	.byte	0x30
	.uleb128 0x4b
	.long	.LASF1693
	.byte	0x53
	.value	0x671
	.long	0xaef7
	.byte	0x38
	.byte	0
	.uleb128 0x37
	.long	.LASF1689
	.byte	0x53
	.value	0x673
	.long	0xaff1
	.uleb128 0x37
	.long	.LASF1694
	.byte	0x53
	.value	0x6e7
	.long	0xb07f
	.uleb128 0x7b
	.long	.LASF1694
	.uleb128 0x36
	.long	.LASF1696
	.byte	0x10
	.byte	0x53
	.value	0x6f6
	.long	0xb0ac
	.uleb128 0x4b
	.long	.LASF1697
	.byte	0x53
	.value	0x6f8
	.long	0x98d1
	.byte	0
	.uleb128 0x4b
	.long	.LASF1666
	.byte	0x53
	.value	0x6f9
	.long	0xb0ac
	.byte	0x8
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0xb084
	.uleb128 0x37
	.long	.LASF1696
	.byte	0x53
	.value	0x6fb
	.long	0xb084
	.uleb128 0x36
	.long	.LASF1698
	.byte	0x10
	.byte	0x53
	.value	0x72b
	.long	0xb0e6
	.uleb128 0x7a
	.string	"len"
	.byte	0x53
	.value	0x72d
	.long	0x30
	.byte	0
	.uleb128 0x7a
	.string	"ptr"
	.byte	0x53
	.value	0x72e
	.long	0x919c
	.byte	0x8
	.byte	0
	.uleb128 0x37
	.long	.LASF1698
	.byte	0x53
	.value	0x730
	.long	0xb0be
	.uleb128 0x37
	.long	.LASF1699
	.byte	0x53
	.value	0x73c
	.long	0xb0fe
	.uleb128 0x7b
	.long	.LASF1700
	.uleb128 0x36
	.long	.LASF1701
	.byte	0x20
	.byte	0x53
	.value	0x73f
	.long	0xb17a
	.uleb128 0x7c
	.byte	0x10
	.byte	0x53
	.value	0x745
	.long	0xb152
	.uleb128 0x7d
	.string	"f"
	.byte	0x53
	.value	0x746
	.long	0x29
	.uleb128 0x7d
	.string	"i"
	.byte	0x53
	.value	0x747
	.long	0x30
	.uleb128 0x7d
	.string	"str"
	.byte	0x53
	.value	0x748
	.long	0xb0e6
	.uleb128 0x7d
	.string	"seq"
	.byte	0x53
	.value	0x749
	.long	0xafeb
	.uleb128 0x7d
	.string	"map"
	.byte	0x53
	.value	0x74a
	.long	0xb17a
	.byte	0
	.uleb128 0x7a
	.string	"tag"
	.byte	0x53
	.value	0x741
	.long	0x30
	.byte	0
	.uleb128 0x4b
	.long	.LASF1702
	.byte	0x53
	.value	0x742
	.long	0xb210
	.byte	0x8
	.uleb128 0x4b
	.long	.LASF209
	.byte	0x53
	.value	0x74b
	.long	0xb110
	.byte	0x10
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0xb0f2
	.uleb128 0x36
	.long	.LASF1703
	.byte	0x48
	.byte	0x53
	.value	0x75c
	.long	0xb210
	.uleb128 0x4b
	.long	.LASF1676
	.byte	0x53
	.value	0x75e
	.long	0x30
	.byte	0
	.uleb128 0x4b
	.long	.LASF1677
	.byte	0x53
	.value	0x75f
	.long	0x30
	.byte	0x4
	.uleb128 0x4b
	.long	.LASF1665
	.byte	0x53
	.value	0x760
	.long	0xb210
	.byte	0x8
	.uleb128 0x4b
	.long	.LASF1666
	.byte	0x53
	.value	0x761
	.long	0xb210
	.byte	0x10
	.uleb128 0x4b
	.long	.LASF1704
	.byte	0x53
	.value	0x762
	.long	0x9472
	.byte	0x18
	.uleb128 0x4b
	.long	.LASF1705
	.byte	0x53
	.value	0x763
	.long	0xb222
	.byte	0x20
	.uleb128 0x4b
	.long	.LASF1706
	.byte	0x53
	.value	0x764
	.long	0xb243
	.byte	0x28
	.uleb128 0x4b
	.long	.LASF1707
	.byte	0x53
	.value	0x765
	.long	0xb261
	.byte	0x30
	.uleb128 0x4b
	.long	.LASF1708
	.byte	0x53
	.value	0x766
	.long	0xb293
	.byte	0x38
	.uleb128 0x4b
	.long	.LASF1709
	.byte	0x53
	.value	0x767
	.long	0xb2c0
	.byte	0x40
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0xb180
	.uleb128 0x37
	.long	.LASF1701
	.byte	0x53
	.value	0x74d
	.long	0xb103
	.uleb128 0x37
	.long	.LASF1710
	.byte	0x53
	.value	0x752
	.long	0xb22e
	.uleb128 0x63
	.byte	0x8
	.long	0xb234
	.uleb128 0x72
	.long	0x30
	.long	0xb243
	.uleb128 0xc
	.long	0xa1f5
	.byte	0
	.uleb128 0x37
	.long	.LASF1711
	.byte	0x53
	.value	0x753
	.long	0xb24f
	.uleb128 0x63
	.byte	0x8
	.long	0xb255
	.uleb128 0x81
	.long	0xb261
	.uleb128 0xc
	.long	0xac71
	.byte	0
	.uleb128 0x37
	.long	.LASF1712
	.byte	0x53
	.value	0x754
	.long	0xb26d
	.uleb128 0x63
	.byte	0x8
	.long	0xb273
	.uleb128 0x72
	.long	0x919a
	.long	0xb287
	.uleb128 0xc
	.long	0xb287
	.uleb128 0xc
	.long	0xb28d
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0xb073
	.uleb128 0x63
	.byte	0x8
	.long	0xb216
	.uleb128 0x37
	.long	.LASF1713
	.byte	0x53
	.value	0x755
	.long	0xb29f
	.uleb128 0x63
	.byte	0x8
	.long	0xb2a5
	.uleb128 0x81
	.long	0xb2c0
	.uleb128 0xc
	.long	0xb287
	.uleb128 0xc
	.long	0x9472
	.uleb128 0xc
	.long	0xa1f5
	.uleb128 0xc
	.long	0xb0b2
	.byte	0
	.uleb128 0x37
	.long	.LASF1714
	.byte	0x53
	.value	0x757
	.long	0xb2cc
	.uleb128 0x63
	.byte	0x8
	.long	0xb2d2
	.uleb128 0x72
	.long	0x919a
	.long	0xb2e1
	.uleb128 0xc
	.long	0xa1f5
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x9ef1
	.uleb128 0x60
	.long	0x9126
	.long	0xb2f9
	.uleb128 0x82
	.long	0x9188
	.value	0x26f
	.byte	0
	.uleb128 0x63
	.byte	0x8
	.long	0x2ba4
	.uleb128 0x63
	.byte	0x8
	.long	0xb305
	.uleb128 0x14
	.long	0x29
	.uleb128 0x6d
	.byte	0x8
	.long	0x2e01
	.uleb128 0x6d
	.byte	0x8
	.long	0x2eba
	.uleb128 0x6d
	.byte	0x8
	.long	0x8303
	.uleb128 0x6d
	.byte	0x8
	.long	0x832f
	.uleb128 0x6d
	.byte	0x8
	.long	0x2f57
	.uleb128 0x6d
	.byte	0x8
	.long	0x2eef
	.uleb128 0x6d
	.byte	0x8
	.long	0x29
	.uleb128 0x6d
	.byte	0x8
	.long	0xb305
	.uleb128 0x63
	.byte	0x8
	.long	0x83ec
	.uleb128 0x6d
	.byte	0x8
	.long	0x8539
	.uleb128 0x63
	.byte	0x8
	.long	0x8539
	.uleb128 0x63
	.byte	0x8
	.long	0x2eef
	.uleb128 0x63
	.byte	0x8
	.long	0x2f68
	.uleb128 0x6d
	.byte	0x8
	.long	0x3024
	.uleb128 0x6f
	.byte	0x8
	.long	0x3019
	.uleb128 0x6d
	.byte	0x8
	.long	0x2f68
	.uleb128 0x6d
	.byte	0x8
	.long	0x3019
	.uleb128 0x63
	.byte	0x8
	.long	0x2f5c
	.uleb128 0x63
	.byte	0x8
	.long	0x3208
	.uleb128 0x6d
	.byte	0x8
	.long	0x30ce
	.uleb128 0x6f
	.byte	0x8
	.long	0x2f5c
	.uleb128 0x63
	.byte	0x8
	.long	0x320d
	.uleb128 0x6d
	.byte	0x8
	.long	0x32f5
	.uleb128 0x6d
	.byte	0x8
	.long	0x3349
	.uleb128 0x6d
	.byte	0x8
	.long	0x3bfa
	.uleb128 0x6f
	.byte	0x8
	.long	0x320d
	.uleb128 0x6d
	.byte	0x8
	.long	0x320d
	.uleb128 0x63
	.byte	0x8
	.long	0x3bfa
	.uleb128 0x6f
	.byte	0x8
	.long	0x3243
	.uleb128 0x63
	.byte	0x8
	.long	0x3bff
	.uleb128 0x63
	.byte	0x8
	.long	0x3ce7
	.uleb128 0x63
	.byte	0x8
	.long	0x853e
	.uleb128 0x6d
	.byte	0x8
	.long	0xb3d0
	.uleb128 0x14
	.long	0xab81
	.uleb128 0x63
	.byte	0x8
	.long	0x876a
	.uleb128 0x6d
	.byte	0x8
	.long	0x853e
	.uleb128 0x83
	.byte	0x20
	.byte	0x40
	.value	0x1aa
	.long	.LASF3075
	.long	0xb40a
	.uleb128 0x4b
	.long	.LASF1715
	.byte	0x40
	.value	0x1ab
	.long	0x9ea0
	.byte	0
	.uleb128 0x4b
	.long	.LASF1716
	.byte	0x40
	.value	0x1ac
	.long	0x9e79
	.byte	0x10
	.byte	0
	.uleb128 0x37
	.long	.LASF1717
	.byte	0x40
	.value	0x1ad
	.long	0xb3e1
	.uleb128 0x2
	.byte	0x8
	.byte	0x3
	.long	.LASF1718
	.uleb128 0x63
	.byte	0x8
	.long	0x3d32
	.uleb128 0x6d
	.byte	0x8
	.long	0x4172
	.uleb128 0x6d
	.byte	0x8
	.long	0x4371
	.uleb128 0x63
	.byte	0x8
	.long	0x4376
	.uleb128 0x6d
	.byte	0x8
	.long	0x3d32
	.uleb128 0x2
	.byte	0x10
	.byte	0x3
	.long	.LASF1719
	.uleb128 0x63
	.byte	0x8
	.long	0x3f1e
	.uleb128 0x6d
	.byte	0x8
	.long	0x4376
	.uleb128 0x63
	.byte	0x8
	.long	0x4172
	.uleb128 0x6d
	.byte	0x8
	.long	0x3f1e
	.uleb128 0x2
	.byte	0x20
	.byte	0x3
	.long	.LASF1720
	.uleb128 0x63
	.byte	0x8
	.long	0x4177
	.uleb128 0x63
	.byte	0x8
	.long	0x4371
	.uleb128 0x6d
	.byte	0x8
	.long	0x4177
	.uleb128 0x4
	.string	"cv"
	.byte	0x54
	.byte	0x43
	.long	0x13f49
	.uleb128 0x23
	.byte	0x54
	.byte	0x4b
	.long	0x1a25
	.uleb128 0x23
	.byte	0x54
	.byte	0x4c
	.long	0x22b2
	.uleb128 0x56
	.long	.LASF1721
	.byte	0x1
	.byte	0x54
	.value	0x1bc
	.uleb128 0x56
	.long	.LASF1722
	.byte	0x1
	.byte	0x54
	.value	0x1bd
	.uleb128 0x56
	.long	.LASF1723
	.byte	0x1
	.byte	0x54
	.value	0x1bf
	.uleb128 0x56
	.long	.LASF1724
	.byte	0x1
	.byte	0x54
	.value	0x1c1
	.uleb128 0x54
	.long	.LASF1725
	.byte	0x8
	.byte	0x54
	.value	0x258
	.long	0xb8e7
	.uleb128 0x34
	.long	0xf2aa
	.byte	0
	.byte	0x1
	.uleb128 0x84
	.string	"Vec"
	.byte	0xd
	.value	0x472
	.long	.LASF1726
	.byte	0x1
	.long	0xb4d9
	.long	0xb4df
	.uleb128 0xb
	.long	0x14320
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0xd
	.value	0x475
	.long	.LASF1727
	.byte	0x1
	.long	0xb4f5
	.long	0xb500
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0xd
	.value	0x479
	.long	.LASF1728
	.byte	0x1
	.long	0xb516
	.long	0xb526
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0xd
	.value	0x47d
	.long	.LASF1729
	.byte	0x1
	.long	0xb53c
	.long	0xb551
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0xd
	.value	0x481
	.long	.LASF1730
	.byte	0x1
	.long	0xb567
	.long	0xb581
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0xd
	.value	0x485
	.long	.LASF1731
	.byte	0x1
	.long	0xb597
	.long	0xb5b6
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0xd
	.value	0x489
	.long	.LASF1732
	.byte	0x1
	.long	0xb5cc
	.long	0xb5f0
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0xd
	.value	0x48d
	.long	.LASF1733
	.byte	0x1
	.long	0xb606
	.long	0xb62f
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0xd
	.value	0x492
	.long	.LASF1734
	.byte	0x1
	.long	0xb645
	.long	0xb673
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0xd
	.value	0x497
	.long	.LASF1735
	.byte	0x1
	.long	0xb689
	.long	0xb6bc
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0xd
	.value	0x49d
	.long	.LASF1736
	.byte	0x1
	.long	0xb6d2
	.long	0xb70a
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x85
	.string	"Vec"
	.byte	0xd
	.value	0x4a3
	.long	.LASF1737
	.byte	0x1
	.long	0xb720
	.long	0xb72b
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x1428f
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0xd
	.value	0x4a8
	.long	.LASF1738
	.byte	0x1
	.long	0xb741
	.long	0xb74c
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x13f5f
	.byte	0
	.uleb128 0x86
	.string	"all"
	.byte	0xd
	.value	0x4bb
	.long	.LASF1739
	.long	0xb4af
	.byte	0x1
	.long	0xb768
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1f
	.string	"mul"
	.byte	0xd
	.value	0x4c2
	.long	.LASF1740
	.long	0xb4af
	.byte	0x1
	.long	0xb781
	.long	0xb78c
	.uleb128 0xb
	.long	0x14326
	.uleb128 0xc
	.long	0x13f5f
	.byte	0
	.uleb128 0x1e
	.long	.LASF1741
	.byte	0x54
	.value	0x275
	.long	.LASF1742
	.long	0xb4af
	.byte	0x1
	.long	0xb7a5
	.long	0xb7ab
	.uleb128 0xb
	.long	0x14326
	.byte	0
	.uleb128 0x1e
	.long	.LASF1743
	.byte	0x54
	.value	0x27c
	.long	.LASF1744
	.long	0xb4af
	.byte	0x1
	.long	0xb7c4
	.long	0xb7cf
	.uleb128 0xb
	.long	0x14326
	.uleb128 0xc
	.long	0x13f5f
	.byte	0
	.uleb128 0x1e
	.long	.LASF1745
	.byte	0x54
	.value	0x280
	.long	.LASF1746
	.long	0xade8
	.byte	0x1
	.long	0xb7e8
	.long	0xb7ee
	.uleb128 0xb
	.long	0x14326
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x54
	.value	0x283
	.long	.LASF1747
	.long	0x14314
	.byte	0x1
	.long	0xb807
	.long	0xb812
	.uleb128 0xb
	.long	0x14326
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x54
	.value	0x284
	.long	.LASF1748
	.long	0x142e6
	.byte	0x1
	.long	0xb82b
	.long	0xb836
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x285
	.long	.LASF1749
	.long	0x14314
	.byte	0x1
	.long	0xb84f
	.long	0xb85a
	.uleb128 0xb
	.long	0x14326
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x286
	.long	.LASF1750
	.long	0x142e6
	.byte	0x1
	.long	0xb873
	.long	0xb87e
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0xd
	.value	0x4ad
	.long	.LASF1751
	.byte	0x1
	.long	0xb894
	.long	0xb8a9
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x1430e
	.uleb128 0xc
	.long	0x1430e
	.uleb128 0xc
	.long	0xb48b
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0xd
	.value	0x4b2
	.long	.LASF1752
	.byte	0x1
	.long	0xb8bf
	.long	0xb8d4
	.uleb128 0xb
	.long	0x14320
	.uleb128 0xc
	.long	0x1430e
	.uleb128 0xc
	.long	0x1430e
	.uleb128 0xc
	.long	0xb494
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x9c79
	.uleb128 0x2e
	.string	"cn"
	.long	0x30
	.byte	0x2
	.byte	0
	.uleb128 0x14
	.long	0xb4af
	.uleb128 0x37
	.long	.LASF1753
	.byte	0x54
	.value	0x377
	.long	0xb8f8
	.uleb128 0x54
	.long	.LASF1754
	.byte	0x10
	.byte	0x54
	.value	0x34e
	.long	0xbafa
	.uleb128 0x87
	.string	"x"
	.byte	0x54
	.value	0x36e
	.long	0x30
	.byte	0
	.byte	0x1
	.uleb128 0x87
	.string	"y"
	.byte	0x54
	.value	0x36e
	.long	0x30
	.byte	0x4
	.byte	0x1
	.uleb128 0x51
	.long	.LASF1631
	.byte	0x54
	.value	0x36e
	.long	0x30
	.byte	0x8
	.byte	0x1
	.uleb128 0x51
	.long	.LASF1632
	.byte	0x54
	.value	0x36e
	.long	0x30
	.byte	0xc
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF1755
	.byte	0xd
	.value	0x76c
	.long	.LASF1756
	.byte	0x1
	.long	0xb950
	.long	0xb956
	.uleb128 0xb
	.long	0x143bc
	.byte	0
	.uleb128 0x1c
	.long	.LASF1755
	.byte	0xd
	.value	0x76d
	.long	.LASF1757
	.byte	0x1
	.long	0xb96b
	.long	0xb985
	.uleb128 0xb
	.long	0x143bc
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF1755
	.byte	0xd
	.value	0x76e
	.long	.LASF1758
	.byte	0x1
	.long	0xb99a
	.long	0xb9a5
	.uleb128 0xb
	.long	0x143bc
	.uleb128 0xc
	.long	0x140f8
	.byte	0
	.uleb128 0x1c
	.long	.LASF1755
	.byte	0xd
	.value	0x76f
	.long	.LASF1759
	.byte	0x1
	.long	0xb9ba
	.long	0xb9c5
	.uleb128 0xb
	.long	0x143bc
	.uleb128 0xc
	.long	0x143c2
	.byte	0
	.uleb128 0x1c
	.long	.LASF1755
	.byte	0xd
	.value	0x770
	.long	.LASF1760
	.byte	0x1
	.long	0xb9da
	.long	0xb9ea
	.uleb128 0xb
	.long	0x143bc
	.uleb128 0xc
	.long	0x13f9e
	.uleb128 0xc
	.long	0x13f98
	.byte	0
	.uleb128 0x1c
	.long	.LASF1755
	.byte	0xd
	.value	0x772
	.long	.LASF1761
	.byte	0x1
	.long	0xb9ff
	.long	0xba0f
	.uleb128 0xb
	.long	0x143bc
	.uleb128 0xc
	.long	0x13f9e
	.uleb128 0xc
	.long	0x13f9e
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0xd
	.value	0x777
	.long	.LASF1762
	.long	0x143cd
	.byte	0x1
	.long	0xba28
	.long	0xba33
	.uleb128 0xb
	.long	0x143bc
	.uleb128 0xc
	.long	0x140f8
	.byte	0
	.uleb128 0x1f
	.string	"tl"
	.byte	0xd
	.value	0x77a
	.long	.LASF1763
	.long	0xc1aa
	.byte	0x1
	.long	0xba4b
	.long	0xba51
	.uleb128 0xb
	.long	0x143d3
	.byte	0
	.uleb128 0x1f
	.string	"br"
	.byte	0xd
	.value	0x77b
	.long	.LASF1764
	.long	0xc1aa
	.byte	0x1
	.long	0xba69
	.long	0xba6f
	.uleb128 0xb
	.long	0x143d3
	.byte	0
	.uleb128 0x1e
	.long	.LASF115
	.byte	0xd
	.value	0x79c
	.long	.LASF1765
	.long	0xc031
	.byte	0x1
	.long	0xba88
	.long	0xba8e
	.uleb128 0xb
	.long	0x143d3
	.byte	0
	.uleb128 0x1e
	.long	.LASF1766
	.byte	0xd
	.value	0x79d
	.long	.LASF1767
	.long	0x30
	.byte	0x1
	.long	0xbaa7
	.long	0xbaad
	.uleb128 0xb
	.long	0x143d3
	.byte	0
	.uleb128 0x1e
	.long	.LASF1768
	.byte	0xd
	.value	0x7a2
	.long	.LASF1769
	.long	0xacb5
	.byte	0x1
	.long	0xbac6
	.long	0xbacc
	.uleb128 0xb
	.long	0x143d3
	.byte	0
	.uleb128 0x1e
	.long	.LASF1770
	.byte	0xd
	.value	0x7a6
	.long	.LASF1771
	.long	0x9ef1
	.byte	0x1
	.long	0xbae5
	.long	0xbaf0
	.uleb128 0xb
	.long	0x143d3
	.uleb128 0xc
	.long	0x13f9e
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x30
	.byte	0
	.uleb128 0x54
	.long	.LASF1772
	.byte	0x8
	.byte	0x54
	.value	0x3c9
	.long	0xbbef
	.uleb128 0x51
	.long	.LASF1773
	.byte	0x54
	.value	0x3d4
	.long	0x30
	.byte	0
	.byte	0x1
	.uleb128 0x87
	.string	"end"
	.byte	0x54
	.value	0x3d4
	.long	0x30
	.byte	0x4
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF1772
	.byte	0x54
	.value	0x3cc
	.long	.LASF1774
	.byte	0x1
	.long	0xbb39
	.long	0xbb3f
	.uleb128 0xb
	.long	0x13f7b
	.byte	0
	.uleb128 0x1c
	.long	.LASF1772
	.byte	0x54
	.value	0x3cd
	.long	.LASF1775
	.byte	0x1
	.long	0xbb54
	.long	0xbb64
	.uleb128 0xb
	.long	0x13f7b
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF1772
	.byte	0x54
	.value	0x3ce
	.long	.LASF1776
	.byte	0x1
	.long	0xbb79
	.long	0xbb84
	.uleb128 0xb
	.long	0x13f7b
	.uleb128 0xc
	.long	0x13f81
	.byte	0
	.uleb128 0x1e
	.long	.LASF115
	.byte	0x54
	.value	0x3cf
	.long	.LASF1777
	.long	0x30
	.byte	0x1
	.long	0xbb9d
	.long	0xbba3
	.uleb128 0xb
	.long	0x13f8c
	.byte	0
	.uleb128 0x1e
	.long	.LASF132
	.byte	0x54
	.value	0x3d0
	.long	.LASF1778
	.long	0x9ef1
	.byte	0x1
	.long	0xbbbc
	.long	0xbbc2
	.uleb128 0xb
	.long	0x13f8c
	.byte	0
	.uleb128 0x4d
	.string	"all"
	.byte	0x54
	.value	0x3d1
	.long	.LASF1779
	.long	0xbafa
	.byte	0x1
	.uleb128 0x55
	.long	.LASF1780
	.byte	0x54
	.value	0x3d2
	.long	.LASF1781
	.long	0xadb1
	.byte	0x1
	.long	0xbbe8
	.uleb128 0xb
	.long	0x13f8c
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0xbafa
	.uleb128 0x54
	.long	.LASF1782
	.byte	0x8
	.byte	0x54
	.value	0x258
	.long	0xc02c
	.uleb128 0x34
	.long	0xe68a
	.byte	0
	.byte	0x1
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x25f
	.long	.LASF1783
	.byte	0x1
	.long	0xbc1e
	.long	0xbc24
	.uleb128 0xb
	.long	0x1411c
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x261
	.long	.LASF1784
	.byte	0x1
	.long	0xbc3a
	.long	0xbc45
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x262
	.long	.LASF1785
	.byte	0x1
	.long	0xbc5b
	.long	0xbc6b
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x263
	.long	.LASF1786
	.byte	0x1
	.long	0xbc81
	.long	0xbc96
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x264
	.long	.LASF1787
	.byte	0x1
	.long	0xbcac
	.long	0xbcc6
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x265
	.long	.LASF1788
	.byte	0x1
	.long	0xbcdc
	.long	0xbcfb
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x266
	.long	.LASF1789
	.byte	0x1
	.long	0xbd11
	.long	0xbd35
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x267
	.long	.LASF1790
	.byte	0x1
	.long	0xbd4b
	.long	0xbd74
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x268
	.long	.LASF1791
	.byte	0x1
	.long	0xbd8a
	.long	0xbdb8
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x269
	.long	.LASF1792
	.byte	0x1
	.long	0xbdce
	.long	0xbe01
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x26a
	.long	.LASF1793
	.byte	0x1
	.long	0xbe17
	.long	0xbe4f
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x85
	.string	"Vec"
	.byte	0x54
	.value	0x26b
	.long	.LASF1794
	.byte	0x1
	.long	0xbe65
	.long	0xbe70
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x9720
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x26d
	.long	.LASF1795
	.byte	0x1
	.long	0xbe86
	.long	0xbe91
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x140e6
	.byte	0
	.uleb128 0x86
	.string	"all"
	.byte	0x54
	.value	0x26f
	.long	.LASF1796
	.long	0xbbf4
	.byte	0x1
	.long	0xbead
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"mul"
	.byte	0x54
	.value	0x272
	.long	.LASF1797
	.long	0xbbf4
	.byte	0x1
	.long	0xbec6
	.long	0xbed1
	.uleb128 0xb
	.long	0x14122
	.uleb128 0xc
	.long	0x140e6
	.byte	0
	.uleb128 0x1e
	.long	.LASF1741
	.byte	0x54
	.value	0x275
	.long	.LASF1798
	.long	0xbbf4
	.byte	0x1
	.long	0xbeea
	.long	0xbef0
	.uleb128 0xb
	.long	0x14122
	.byte	0
	.uleb128 0x1e
	.long	.LASF1743
	.byte	0x54
	.value	0x27c
	.long	.LASF1799
	.long	0xbbf4
	.byte	0x1
	.long	0xbf09
	.long	0xbf14
	.uleb128 0xb
	.long	0x14122
	.uleb128 0xc
	.long	0x140e6
	.byte	0
	.uleb128 0x1e
	.long	.LASF1745
	.byte	0x54
	.value	0x280
	.long	.LASF1800
	.long	0xade8
	.byte	0x1
	.long	0xbf2d
	.long	0xbf33
	.uleb128 0xb
	.long	0x14122
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x54
	.value	0x283
	.long	.LASF1801
	.long	0x13fbc
	.byte	0x1
	.long	0xbf4c
	.long	0xbf57
	.uleb128 0xb
	.long	0x14122
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x54
	.value	0x284
	.long	.LASF1802
	.long	0x13fc2
	.byte	0x1
	.long	0xbf70
	.long	0xbf7b
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x285
	.long	.LASF1803
	.long	0x13fbc
	.byte	0x1
	.long	0xbf94
	.long	0xbf9f
	.uleb128 0xb
	.long	0x14122
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x286
	.long	.LASF1804
	.long	0x13fc2
	.byte	0x1
	.long	0xbfb8
	.long	0xbfc3
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x288
	.long	.LASF1805
	.byte	0x1
	.long	0xbfd9
	.long	0xbfee
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x14110
	.uleb128 0xc
	.long	0x14110
	.uleb128 0xc
	.long	0xb48b
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x289
	.long	.LASF1806
	.byte	0x1
	.long	0xc004
	.long	0xc019
	.uleb128 0xb
	.long	0x1411c
	.uleb128 0xc
	.long	0x14110
	.uleb128 0xc
	.long	0x14110
	.uleb128 0xc
	.long	0xb494
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x30
	.uleb128 0x2e
	.string	"cn"
	.long	0x30
	.byte	0x2
	.byte	0
	.uleb128 0x2a
	.long	.LASF1807
	.uleb128 0x54
	.long	.LASF1808
	.byte	0x8
	.byte	0x54
	.value	0x32b
	.long	0xc1a5
	.uleb128 0x51
	.long	.LASF1631
	.byte	0x54
	.value	0x343
	.long	0x30
	.byte	0
	.byte	0x1
	.uleb128 0x51
	.long	.LASF1632
	.byte	0x54
	.value	0x343
	.long	0x30
	.byte	0x4
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF1809
	.byte	0x54
	.value	0x331
	.long	.LASF1810
	.byte	0x1
	.long	0xc06f
	.long	0xc075
	.uleb128 0xb
	.long	0x13f92
	.byte	0
	.uleb128 0x1c
	.long	.LASF1809
	.byte	0x54
	.value	0x332
	.long	.LASF1811
	.byte	0x1
	.long	0xc08a
	.long	0xc09a
	.uleb128 0xb
	.long	0x13f92
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF1809
	.byte	0x54
	.value	0x333
	.long	.LASF1812
	.byte	0x1
	.long	0xc0af
	.long	0xc0ba
	.uleb128 0xb
	.long	0x13f92
	.uleb128 0xc
	.long	0x13f98
	.byte	0
	.uleb128 0x1c
	.long	.LASF1809
	.byte	0x54
	.value	0x334
	.long	.LASF1813
	.byte	0x1
	.long	0xc0cf
	.long	0xc0da
	.uleb128 0xb
	.long	0x13f92
	.uleb128 0xc
	.long	0x13f65
	.byte	0
	.uleb128 0x1c
	.long	.LASF1809
	.byte	0x54
	.value	0x335
	.long	.LASF1814
	.byte	0x1
	.long	0xc0ef
	.long	0xc0fa
	.uleb128 0xb
	.long	0x13f92
	.uleb128 0xc
	.long	0x13f70
	.byte	0
	.uleb128 0x1c
	.long	.LASF1809
	.byte	0x54
	.value	0x336
	.long	.LASF1815
	.byte	0x1
	.long	0xc10f
	.long	0xc11a
	.uleb128 0xb
	.long	0x13f92
	.uleb128 0xc
	.long	0x13f9e
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0x54
	.value	0x338
	.long	.LASF1816
	.long	0x13fa4
	.byte	0x1
	.long	0xc133
	.long	0xc13e
	.uleb128 0xb
	.long	0x13f92
	.uleb128 0xc
	.long	0x13f98
	.byte	0
	.uleb128 0x1e
	.long	.LASF1766
	.byte	0x54
	.value	0x33a
	.long	.LASF1817
	.long	0x30
	.byte	0x1
	.long	0xc157
	.long	0xc15d
	.uleb128 0xb
	.long	0x13faa
	.byte	0
	.uleb128 0x1e
	.long	.LASF1818
	.byte	0x54
	.value	0x340
	.long	.LASF1819
	.long	0xad49
	.byte	0x1
	.long	0xc176
	.long	0xc17c
	.uleb128 0xb
	.long	0x13faa
	.byte	0
	.uleb128 0x1e
	.long	.LASF1820
	.byte	0x54
	.value	0x341
	.long	.LASF1821
	.long	0xad7d
	.byte	0x1
	.long	0xc195
	.long	0xc19b
	.uleb128 0xb
	.long	0x13faa
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x30
	.byte	0
	.uleb128 0x14
	.long	0xc031
	.uleb128 0x54
	.long	.LASF1822
	.byte	0x8
	.byte	0x54
	.value	0x2d9
	.long	0xc3cc
	.uleb128 0x87
	.string	"x"
	.byte	0x54
	.value	0x2f9
	.long	0x30
	.byte	0
	.byte	0x1
	.uleb128 0x87
	.string	"y"
	.byte	0x54
	.value	0x2f9
	.long	0x30
	.byte	0x4
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF1823
	.byte	0x54
	.value	0x2df
	.long	.LASF1824
	.byte	0x1
	.long	0xc1e6
	.long	0xc1ec
	.uleb128 0xb
	.long	0x140e0
	.byte	0
	.uleb128 0x1c
	.long	.LASF1823
	.byte	0x54
	.value	0x2e0
	.long	.LASF1825
	.byte	0x1
	.long	0xc201
	.long	0xc211
	.uleb128 0xb
	.long	0x140e0
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF1823
	.byte	0x54
	.value	0x2e1
	.long	.LASF1826
	.byte	0x1
	.long	0xc226
	.long	0xc231
	.uleb128 0xb
	.long	0x140e0
	.uleb128 0xc
	.long	0x13f9e
	.byte	0
	.uleb128 0x1c
	.long	.LASF1823
	.byte	0x54
	.value	0x2e2
	.long	.LASF1827
	.byte	0x1
	.long	0xc246
	.long	0xc251
	.uleb128 0xb
	.long	0x140e0
	.uleb128 0xc
	.long	0x13f49
	.byte	0
	.uleb128 0x1c
	.long	.LASF1823
	.byte	0x54
	.value	0x2e3
	.long	.LASF1828
	.byte	0x1
	.long	0xc266
	.long	0xc271
	.uleb128 0xb
	.long	0x140e0
	.uleb128 0xc
	.long	0x13f54
	.byte	0
	.uleb128 0x1c
	.long	.LASF1823
	.byte	0x54
	.value	0x2e4
	.long	.LASF1829
	.byte	0x1
	.long	0xc286
	.long	0xc291
	.uleb128 0xb
	.long	0x140e0
	.uleb128 0xc
	.long	0x13f98
	.byte	0
	.uleb128 0x1c
	.long	.LASF1823
	.byte	0x54
	.value	0x2e5
	.long	.LASF1830
	.byte	0x1
	.long	0xc2a6
	.long	0xc2b1
	.uleb128 0xb
	.long	0x140e0
	.uleb128 0xc
	.long	0x140e6
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0x54
	.value	0x2e7
	.long	.LASF1831
	.long	0x140ec
	.byte	0x1
	.long	0xc2ca
	.long	0xc2d5
	.uleb128 0xb
	.long	0x140e0
	.uleb128 0xc
	.long	0x13f9e
	.byte	0
	.uleb128 0x1e
	.long	.LASF1832
	.byte	0x54
	.value	0x2ec
	.long	.LASF1833
	.long	0xace5
	.byte	0x1
	.long	0xc2ee
	.long	0xc2f4
	.uleb128 0xb
	.long	0x140f2
	.byte	0
	.uleb128 0x1e
	.long	.LASF1834
	.byte	0x54
	.value	0x2ed
	.long	.LASF1835
	.long	0xad15
	.byte	0x1
	.long	0xc30d
	.long	0xc313
	.uleb128 0xb
	.long	0x140f2
	.byte	0
	.uleb128 0x1e
	.long	.LASF1836
	.byte	0x54
	.value	0x2ee
	.long	.LASF1837
	.long	0xbbf4
	.byte	0x1
	.long	0xc32c
	.long	0xc332
	.uleb128 0xb
	.long	0x140f2
	.byte	0
	.uleb128 0x1f
	.string	"dot"
	.byte	0x54
	.value	0x2f1
	.long	.LASF1838
	.long	0x30
	.byte	0x1
	.long	0xc34b
	.long	0xc356
	.uleb128 0xb
	.long	0x140f2
	.uleb128 0xc
	.long	0x13f9e
	.byte	0
	.uleb128 0x1e
	.long	.LASF1839
	.byte	0x54
	.value	0x2f3
	.long	.LASF1840
	.long	0x29
	.byte	0x1
	.long	0xc36f
	.long	0xc37a
	.uleb128 0xb
	.long	0x140f2
	.uleb128 0xc
	.long	0x13f9e
	.byte	0
	.uleb128 0x1e
	.long	.LASF1743
	.byte	0x54
	.value	0x2f5
	.long	.LASF1841
	.long	0x29
	.byte	0x1
	.long	0xc393
	.long	0xc39e
	.uleb128 0xb
	.long	0x140f2
	.uleb128 0xc
	.long	0x13f9e
	.byte	0
	.uleb128 0x1e
	.long	.LASF1842
	.byte	0x54
	.value	0x2f7
	.long	.LASF1843
	.long	0x9ef1
	.byte	0x1
	.long	0xc3b7
	.long	0xc3c2
	.uleb128 0xb
	.long	0x140f2
	.uleb128 0xc
	.long	0x140f8
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x30
	.byte	0
	.uleb128 0x14
	.long	0xc1aa
	.uleb128 0x88
	.byte	0x4
	.long	0x913b
	.byte	0x54
	.value	0x5c8
	.long	0xc3f6
	.uleb128 0x3c
	.long	.LASF1844
	.long	0xffff0000
	.uleb128 0x3b
	.long	.LASF1845
	.value	0xfff
	.uleb128 0x10
	.long	.LASF1846
	.byte	0x7
	.byte	0
	.uleb128 0x89
	.string	"Mat"
	.byte	0x60
	.byte	0x54
	.value	0x6ac
	.long	0xd5e0
	.uleb128 0x8a
	.byte	0x4
	.long	0x913b
	.byte	0x54
	.value	0x7b2
	.byte	0x1
	.long	0xc431
	.uleb128 0x3c
	.long	.LASF1847
	.long	0x42ff0000
	.uleb128 0x10
	.long	.LASF1848
	.byte	0
	.uleb128 0x3b
	.long	.LASF1849
	.value	0x4000
	.uleb128 0x3b
	.long	.LASF1850
	.value	0x8000
	.byte	0
	.uleb128 0x8b
	.long	.LASF1851
	.byte	0x8
	.byte	0x54
	.value	0x7ce
	.byte	0x1
	.long	0xc52f
	.uleb128 0x7a
	.string	"p"
	.byte	0x54
	.value	0x7d8
	.long	0xaa8a
	.byte	0
	.uleb128 0x19
	.long	.LASF1851
	.byte	0x54
	.value	0x7d0
	.long	.LASF1852
	.long	0xc45f
	.long	0xc46a
	.uleb128 0xb
	.long	0x13fb0
	.uleb128 0xc
	.long	0xaa8a
	.byte	0
	.uleb128 0x18
	.long	.LASF480
	.byte	0x54
	.value	0x7d1
	.long	.LASF1853
	.long	0xd5e0
	.long	0xc482
	.long	0xc488
	.uleb128 0xb
	.long	0x13fb6
	.byte	0
	.uleb128 0x18
	.long	.LASF134
	.byte	0x54
	.value	0x7d2
	.long	.LASF1854
	.long	0x13fbc
	.long	0xc4a0
	.long	0xc4ab
	.uleb128 0xb
	.long	0x13fb6
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x18
	.long	.LASF134
	.byte	0x54
	.value	0x7d3
	.long	.LASF1855
	.long	0x13fc2
	.long	0xc4c3
	.long	0xc4ce
	.uleb128 0xb
	.long	0x13fb0
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x18
	.long	.LASF1856
	.byte	0x54
	.value	0x7d4
	.long	.LASF1857
	.long	0x9720
	.long	0xc4e6
	.long	0xc4ec
	.uleb128 0xb
	.long	0x13fb6
	.byte	0
	.uleb128 0x18
	.long	.LASF1858
	.byte	0x54
	.value	0x7d5
	.long	.LASF1859
	.long	0x9ef1
	.long	0xc504
	.long	0xc50f
	.uleb128 0xb
	.long	0x13fb6
	.uleb128 0xc
	.long	0x13fc8
	.byte	0
	.uleb128 0x53
	.long	.LASF1860
	.byte	0x54
	.value	0x7d6
	.long	.LASF1861
	.long	0x9ef1
	.long	0xc523
	.uleb128 0xb
	.long	0x13fb6
	.uleb128 0xc
	.long	0x13fc8
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0xc431
	.uleb128 0x8b
	.long	.LASF1862
	.byte	0x18
	.byte	0x54
	.value	0x7db
	.byte	0x1
	.long	0xc63c
	.uleb128 0x7a
	.string	"p"
	.byte	0x54
	.value	0x7e4
	.long	0x13fce
	.byte	0
	.uleb128 0x7a
	.string	"buf"
	.byte	0x54
	.value	0x7e5
	.long	0x13fd4
	.byte	0x8
	.uleb128 0x19
	.long	.LASF1862
	.byte	0x54
	.value	0x7dd
	.long	.LASF1863
	.long	0xc56f
	.long	0xc575
	.uleb128 0xb
	.long	0x13fe4
	.byte	0
	.uleb128 0x19
	.long	.LASF1862
	.byte	0x54
	.value	0x7de
	.long	.LASF1864
	.long	0xc589
	.long	0xc594
	.uleb128 0xb
	.long	0x13fe4
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x18
	.long	.LASF134
	.byte	0x54
	.value	0x7df
	.long	.LASF1865
	.long	0x13fea
	.long	0xc5ac
	.long	0xc5b7
	.uleb128 0xb
	.long	0x13ff5
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x18
	.long	.LASF134
	.byte	0x54
	.value	0x7e0
	.long	.LASF1866
	.long	0x13ffb
	.long	0xc5cf
	.long	0xc5da
	.uleb128 0xb
	.long	0x13fe4
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x18
	.long	.LASF1867
	.byte	0x54
	.value	0x7e1
	.long	.LASF1868
	.long	0x911b
	.long	0xc5f2
	.long	0xc5f8
	.uleb128 0xb
	.long	0x13ff5
	.byte	0
	.uleb128 0x18
	.long	.LASF89
	.byte	0x54
	.value	0x7e2
	.long	.LASF1869
	.long	0x14001
	.long	0xc610
	.long	0xc61b
	.uleb128 0xb
	.long	0x13fe4
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x55
	.long	.LASF89
	.byte	0x54
	.value	0x7e7
	.long	.LASF1870
	.long	0x14001
	.byte	0x2
	.long	0xc630
	.uleb128 0xb
	.long	0x13fe4
	.uleb128 0xc
	.long	0x14007
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0xc534
	.uleb128 0x51
	.long	.LASF1676
	.byte	0x54
	.value	0x7ba
	.long	0x30
	.byte	0
	.byte	0x1
	.uleb128 0x51
	.long	.LASF1654
	.byte	0x54
	.value	0x7bc
	.long	0x30
	.byte	0x4
	.byte	0x1
	.uleb128 0x51
	.long	.LASF1647
	.byte	0x54
	.value	0x7be
	.long	0x30
	.byte	0x8
	.byte	0x1
	.uleb128 0x51
	.long	.LASF1648
	.byte	0x54
	.value	0x7be
	.long	0x30
	.byte	0xc
	.byte	0x1
	.uleb128 0x51
	.long	.LASF209
	.byte	0x54
	.value	0x7c0
	.long	0xab75
	.byte	0x10
	.byte	0x1
	.uleb128 0x51
	.long	.LASF1651
	.byte	0x54
	.value	0x7c4
	.long	0xaa8a
	.byte	0x18
	.byte	0x1
	.uleb128 0x51
	.long	.LASF1871
	.byte	0x54
	.value	0x7c7
	.long	0xab75
	.byte	0x20
	.byte	0x1
	.uleb128 0x51
	.long	.LASF1872
	.byte	0x54
	.value	0x7c8
	.long	0xab75
	.byte	0x28
	.byte	0x1
	.uleb128 0x51
	.long	.LASF1873
	.byte	0x54
	.value	0x7c9
	.long	0xab75
	.byte	0x30
	.byte	0x1
	.uleb128 0x51
	.long	.LASF328
	.byte	0x54
	.value	0x7cc
	.long	0x1400d
	.byte	0x38
	.byte	0x1
	.uleb128 0x51
	.long	.LASF115
	.byte	0x54
	.value	0x7ea
	.long	0xc431
	.byte	0x40
	.byte	0x1
	.uleb128 0x51
	.long	.LASF1650
	.byte	0x54
	.value	0x7eb
	.long	0xc534
	.byte	0x48
	.byte	0x1
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6b0
	.long	.LASF1874
	.byte	0x1
	.long	0xc6ff
	.long	0xc705
	.uleb128 0xb
	.long	0x14013
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6b3
	.long	.LASF1875
	.byte	0x1
	.long	0xc71b
	.long	0xc730
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6b4
	.long	.LASF1876
	.byte	0x1
	.long	0xc746
	.long	0xc756
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0xd5e0
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6b6
	.long	.LASF1877
	.byte	0x1
	.long	0xc76c
	.long	0xc786
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x14019
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6b7
	.long	.LASF1878
	.byte	0x1
	.long	0xc79c
	.long	0xc7b1
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0xd5e0
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x14019
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6ba
	.long	.LASF1879
	.byte	0x1
	.long	0xc7c7
	.long	0xc7dc
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x9720
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6bb
	.long	.LASF1880
	.byte	0x1
	.long	0xc7f2
	.long	0xc80c
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x9720
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x14019
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6be
	.long	.LASF1881
	.byte	0x1
	.long	0xc822
	.long	0xc82d
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x1401f
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6c0
	.long	.LASF1882
	.byte	0x1
	.long	0xc843
	.long	0xc862
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x919a
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6c1
	.long	.LASF1883
	.byte	0x1
	.long	0xc878
	.long	0xc892
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0xd5e0
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x919a
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6c2
	.long	.LASF1884
	.byte	0x1
	.long	0xc8a8
	.long	0xc8c7
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x9720
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x919a
	.uleb128 0xc
	.long	0x14025
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6c5
	.long	.LASF1885
	.byte	0x1
	.long	0xc8dd
	.long	0xc8f2
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x1401f
	.uleb128 0xc
	.long	0x1402b
	.uleb128 0xc
	.long	0x1402b
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6c6
	.long	.LASF1886
	.byte	0x1
	.long	0xc908
	.long	0xc918
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x1401f
	.uleb128 0xc
	.long	0x14031
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6c7
	.long	.LASF1887
	.byte	0x1
	.long	0xc92e
	.long	0xc93e
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x1401f
	.uleb128 0xc
	.long	0x13f8c
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6c9
	.long	.LASF1888
	.byte	0x1
	.long	0xc954
	.long	0xc964
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x14037
	.uleb128 0xc
	.long	0x9ef1
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6cb
	.long	.LASF1889
	.byte	0x1
	.long	0xc97a
	.long	0xc98a
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x14042
	.uleb128 0xc
	.long	0x9ef1
	.byte	0
	.uleb128 0x84
	.string	"Mat"
	.byte	0x54
	.value	0x6cd
	.long	.LASF1890
	.byte	0x1
	.long	0xc9a0
	.long	0xc9b0
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x1404d
	.uleb128 0xc
	.long	0x9ef1
	.byte	0
	.uleb128 0x85
	.string	"Mat"
	.byte	0x54
	.value	0x6dc
	.long	.LASF1891
	.byte	0x1
	.long	0xc9c6
	.long	0xc9d1
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x14058
	.byte	0
	.uleb128 0x1c
	.long	.LASF1892
	.byte	0x54
	.value	0x6df
	.long	.LASF1893
	.byte	0x1
	.long	0xc9e6
	.long	0xc9f1
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0x54
	.value	0x6e1
	.long	.LASF1894
	.long	0x1405e
	.byte	0x1
	.long	0xca0a
	.long	0xca15
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x1401f
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0x54
	.value	0x6e2
	.long	.LASF1895
	.long	0x1405e
	.byte	0x1
	.long	0xca2e
	.long	0xca39
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x14064
	.byte	0
	.uleb128 0x1f
	.string	"row"
	.byte	0x54
	.value	0x6e5
	.long	.LASF1896
	.long	0xc3f6
	.byte	0x1
	.long	0xca52
	.long	0xca5d
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"col"
	.byte	0x54
	.value	0x6e7
	.long	.LASF1897
	.long	0xc3f6
	.byte	0x1
	.long	0xca76
	.long	0xca81
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1898
	.byte	0x54
	.value	0x6e9
	.long	.LASF1899
	.long	0xc3f6
	.byte	0x1
	.long	0xca9a
	.long	0xcaaa
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1898
	.byte	0x54
	.value	0x6ea
	.long	.LASF1900
	.long	0xc3f6
	.byte	0x1
	.long	0xcac3
	.long	0xcace
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x1402b
	.byte	0
	.uleb128 0x1e
	.long	.LASF1901
	.byte	0x54
	.value	0x6ec
	.long	.LASF1902
	.long	0xc3f6
	.byte	0x1
	.long	0xcae7
	.long	0xcaf7
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1901
	.byte	0x54
	.value	0x6ed
	.long	.LASF1903
	.long	0xc3f6
	.byte	0x1
	.long	0xcb10
	.long	0xcb1b
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x1402b
	.byte	0
	.uleb128 0x1e
	.long	.LASF1904
	.byte	0x54
	.value	0x6f2
	.long	.LASF1905
	.long	0xc3f6
	.byte	0x1
	.long	0xcb34
	.long	0xcb3f
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x8c
	.long	.LASF1904
	.byte	0x54
	.value	0x6f4
	.long	.LASF1906
	.long	0xc3f6
	.byte	0x1
	.long	0xcb5b
	.uleb128 0xc
	.long	0x1401f
	.byte	0
	.uleb128 0x1e
	.long	.LASF1709
	.byte	0x54
	.value	0x6f7
	.long	.LASF1907
	.long	0xc3f6
	.byte	0x1
	.long	0xcb74
	.long	0xcb7a
	.uleb128 0xb
	.long	0x1406a
	.byte	0
	.uleb128 0x1c
	.long	.LASF1908
	.byte	0x54
	.value	0x6fa
	.long	.LASF1909
	.byte	0x1
	.long	0xcb8f
	.long	0xcb9a
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0xdb57
	.byte	0
	.uleb128 0x1c
	.long	.LASF1908
	.byte	0x54
	.value	0x6fc
	.long	.LASF1910
	.byte	0x1
	.long	0xcbaf
	.long	0xcbbf
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0xdb57
	.uleb128 0xc
	.long	0xdb6d
	.byte	0
	.uleb128 0x1c
	.long	.LASF1911
	.byte	0x54
	.value	0x6fe
	.long	.LASF1912
	.byte	0x1
	.long	0xcbd4
	.long	0xcbee
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0xdb57
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF1913
	.byte	0x54
	.value	0x700
	.long	.LASF1914
	.byte	0x1
	.long	0xcc03
	.long	0xcc13
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x1405e
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0x54
	.value	0x703
	.long	.LASF1915
	.long	0x1405e
	.byte	0x1
	.long	0xcc2c
	.long	0xcc37
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x14019
	.byte	0
	.uleb128 0x1e
	.long	.LASF1916
	.byte	0x54
	.value	0x705
	.long	.LASF1917
	.long	0x1405e
	.byte	0x1
	.long	0xcc50
	.long	0xcc60
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0xdb6d
	.uleb128 0xc
	.long	0xdb6d
	.byte	0
	.uleb128 0x1e
	.long	.LASF1918
	.byte	0x54
	.value	0x708
	.long	.LASF1919
	.long	0xc3f6
	.byte	0x1
	.long	0xcc79
	.long	0xcc89
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1918
	.byte	0x54
	.value	0x709
	.long	.LASF1920
	.long	0xc3f6
	.byte	0x1
	.long	0xcca2
	.long	0xccb7
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x9720
	.byte	0
	.uleb128 0x1f
	.string	"t"
	.byte	0x54
	.value	0x70c
	.long	.LASF1921
	.long	0xd849
	.byte	0x1
	.long	0xccce
	.long	0xccd4
	.uleb128 0xb
	.long	0x1406a
	.byte	0
	.uleb128 0x1f
	.string	"inv"
	.byte	0x54
	.value	0x70e
	.long	.LASF1922
	.long	0xd849
	.byte	0x1
	.long	0xcced
	.long	0xccf8
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"mul"
	.byte	0x54
	.value	0x710
	.long	.LASF1923
	.long	0xd849
	.byte	0x1
	.long	0xcd11
	.long	0xcd21
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0xdb6d
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1e
	.long	.LASF1743
	.byte	0x54
	.value	0x713
	.long	.LASF1924
	.long	0xc3f6
	.byte	0x1
	.long	0xcd3a
	.long	0xcd45
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0xdb6d
	.byte	0
	.uleb128 0x1f
	.string	"dot"
	.byte	0x54
	.value	0x715
	.long	.LASF1925
	.long	0x29
	.byte	0x1
	.long	0xcd5e
	.long	0xcd69
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0xdb6d
	.byte	0
	.uleb128 0x8c
	.long	.LASF1926
	.byte	0x54
	.value	0x718
	.long	.LASF1927
	.long	0xd849
	.byte	0x1
	.long	0xcd8f
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x8c
	.long	.LASF1926
	.byte	0x54
	.value	0x719
	.long	.LASF1928
	.long	0xd849
	.byte	0x1
	.long	0xcdb0
	.uleb128 0xc
	.long	0xd5e0
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x8c
	.long	.LASF1926
	.byte	0x54
	.value	0x71a
	.long	.LASF1929
	.long	0xd849
	.byte	0x1
	.long	0xcdd6
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x9720
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x8c
	.long	.LASF1930
	.byte	0x54
	.value	0x71b
	.long	.LASF1931
	.long	0xd849
	.byte	0x1
	.long	0xcdfc
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x8c
	.long	.LASF1930
	.byte	0x54
	.value	0x71c
	.long	.LASF1932
	.long	0xd849
	.byte	0x1
	.long	0xce1d
	.uleb128 0xc
	.long	0xd5e0
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x8c
	.long	.LASF1930
	.byte	0x54
	.value	0x71d
	.long	.LASF1933
	.long	0xd849
	.byte	0x1
	.long	0xce43
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x9720
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x86
	.string	"eye"
	.byte	0x54
	.value	0x71e
	.long	.LASF1934
	.long	0xd849
	.byte	0x1
	.long	0xce69
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x86
	.string	"eye"
	.byte	0x54
	.value	0x71f
	.long	.LASF1935
	.long	0xd849
	.byte	0x1
	.long	0xce8a
	.uleb128 0xc
	.long	0xd5e0
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF1936
	.byte	0x54
	.value	0x723
	.long	.LASF1937
	.byte	0x1
	.long	0xce9f
	.long	0xceb4
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF1936
	.byte	0x54
	.value	0x724
	.long	.LASF1938
	.byte	0x1
	.long	0xcec9
	.long	0xced9
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0xd5e0
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF1936
	.byte	0x54
	.value	0x725
	.long	.LASF1939
	.byte	0x1
	.long	0xceee
	.long	0xcf03
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x9720
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF1940
	.byte	0x54
	.value	0x728
	.long	.LASF1941
	.byte	0x1
	.long	0xcf18
	.long	0xcf1e
	.uleb128 0xb
	.long	0x14013
	.byte	0
	.uleb128 0x1c
	.long	.LASF1706
	.byte	0x54
	.value	0x72b
	.long	.LASF1942
	.byte	0x1
	.long	0xcf33
	.long	0xcf39
	.uleb128 0xb
	.long	0x14013
	.byte	0
	.uleb128 0x1c
	.long	.LASF338
	.byte	0x54
	.value	0x72e
	.long	.LASF1943
	.byte	0x1
	.long	0xcf4e
	.long	0xcf54
	.uleb128 0xb
	.long	0x14013
	.byte	0
	.uleb128 0x1c
	.long	.LASF1944
	.byte	0x54
	.value	0x730
	.long	.LASF1945
	.byte	0x1
	.long	0xcf69
	.long	0xcf74
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x1401f
	.byte	0
	.uleb128 0x1c
	.long	.LASF128
	.byte	0x54
	.value	0x733
	.long	.LASF1946
	.byte	0x1
	.long	0xcf89
	.long	0xcf94
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x1c
	.long	.LASF121
	.byte	0x54
	.value	0x735
	.long	.LASF1947
	.byte	0x1
	.long	0xcfa9
	.long	0xcfb4
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x1c
	.long	.LASF121
	.byte	0x54
	.value	0x737
	.long	.LASF1948
	.byte	0x1
	.long	0xcfc9
	.long	0xcfd9
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x911b
	.uleb128 0xc
	.long	0x14019
	.byte	0
	.uleb128 0x1c
	.long	.LASF1949
	.byte	0x54
	.value	0x739
	.long	.LASF1950
	.byte	0x1
	.long	0xcfee
	.long	0xcff9
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0xa1f5
	.byte	0
	.uleb128 0x1c
	.long	.LASF157
	.byte	0x54
	.value	0x73d
	.long	.LASF1951
	.byte	0x1
	.long	0xd00e
	.long	0xd019
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x1401f
	.byte	0
	.uleb128 0x1c
	.long	.LASF180
	.byte	0x54
	.value	0x73f
	.long	.LASF1952
	.byte	0x1
	.long	0xd02e
	.long	0xd039
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x1c
	.long	.LASF1953
	.byte	0x54
	.value	0x742
	.long	.LASF1954
	.byte	0x1
	.long	0xd04e
	.long	0xd05e
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x1407c
	.uleb128 0xc
	.long	0x14082
	.byte	0
	.uleb128 0x1e
	.long	.LASF1955
	.byte	0x54
	.value	0x744
	.long	.LASF1956
	.long	0x1405e
	.byte	0x1
	.long	0xd077
	.long	0xd091
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x747
	.long	.LASF1957
	.long	0xc3f6
	.byte	0x1
	.long	0xd0aa
	.long	0xd0ba
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0xbafa
	.uleb128 0xc
	.long	0xbafa
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x748
	.long	.LASF1958
	.long	0xc3f6
	.byte	0x1
	.long	0xd0d3
	.long	0xd0de
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x14031
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x749
	.long	.LASF1959
	.long	0xc3f6
	.byte	0x1
	.long	0xd0f7
	.long	0xd102
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x13f8c
	.byte	0
	.uleb128 0x1e
	.long	.LASF1960
	.byte	0x54
	.value	0x74c
	.long	.LASF1961
	.long	0xab87
	.byte	0x1
	.long	0xd11b
	.long	0xd121
	.uleb128 0xb
	.long	0x1406a
	.byte	0
	.uleb128 0x1e
	.long	.LASF1962
	.byte	0x54
	.value	0x74e
	.long	.LASF1963
	.long	0xac65
	.byte	0x1
	.long	0xd13a
	.long	0xd140
	.uleb128 0xb
	.long	0x1406a
	.byte	0
	.uleb128 0x1e
	.long	.LASF1964
	.byte	0x54
	.value	0x750
	.long	.LASF1965
	.long	0xaa7e
	.byte	0x1
	.long	0xd159
	.long	0xd15f
	.uleb128 0xb
	.long	0x1406a
	.byte	0
	.uleb128 0x1e
	.long	.LASF1966
	.byte	0x54
	.value	0x759
	.long	.LASF1967
	.long	0x9ef1
	.byte	0x1
	.long	0xd178
	.long	0xd17e
	.uleb128 0xb
	.long	0x1406a
	.byte	0
	.uleb128 0x1e
	.long	.LASF1968
	.byte	0x54
	.value	0x75c
	.long	.LASF1969
	.long	0x9ef1
	.byte	0x1
	.long	0xd197
	.long	0xd19d
	.uleb128 0xb
	.long	0x1406a
	.byte	0
	.uleb128 0x1e
	.long	.LASF1970
	.byte	0x54
	.value	0x760
	.long	.LASF1971
	.long	0x911b
	.byte	0x1
	.long	0xd1b6
	.long	0xd1bc
	.uleb128 0xb
	.long	0x1406a
	.byte	0
	.uleb128 0x1e
	.long	.LASF1972
	.byte	0x54
	.value	0x762
	.long	.LASF1973
	.long	0x911b
	.byte	0x1
	.long	0xd1d5
	.long	0xd1db
	.uleb128 0xb
	.long	0x1406a
	.byte	0
	.uleb128 0x1e
	.long	.LASF1649
	.byte	0x54
	.value	0x764
	.long	.LASF1974
	.long	0x30
	.byte	0x1
	.long	0xd1f4
	.long	0xd1fa
	.uleb128 0xb
	.long	0x1406a
	.byte	0
	.uleb128 0x1e
	.long	.LASF1625
	.byte	0x54
	.value	0x766
	.long	.LASF1975
	.long	0x30
	.byte	0x1
	.long	0xd213
	.long	0xd219
	.uleb128 0xb
	.long	0x1406a
	.byte	0
	.uleb128 0x1e
	.long	.LASF1976
	.byte	0x54
	.value	0x768
	.long	.LASF1977
	.long	0x30
	.byte	0x1
	.long	0xd232
	.long	0xd238
	.uleb128 0xb
	.long	0x1406a
	.byte	0
	.uleb128 0x1e
	.long	.LASF1978
	.byte	0x54
	.value	0x76a
	.long	.LASF1979
	.long	0x911b
	.byte	0x1
	.long	0xd251
	.long	0xd25c
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF132
	.byte	0x54
	.value	0x76c
	.long	.LASF1980
	.long	0x9ef1
	.byte	0x1
	.long	0xd275
	.long	0xd27b
	.uleb128 0xb
	.long	0x1406a
	.byte	0
	.uleb128 0x1e
	.long	.LASF1682
	.byte	0x54
	.value	0x76e
	.long	.LASF1981
	.long	0x911b
	.byte	0x1
	.long	0xd294
	.long	0xd29a
	.uleb128 0xb
	.long	0x1406a
	.byte	0
	.uleb128 0x1e
	.long	.LASF1982
	.byte	0x54
	.value	0x771
	.long	.LASF1983
	.long	0x30
	.byte	0x1
	.long	0xd2b3
	.long	0xd2c8
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x9ef1
	.byte	0
	.uleb128 0x1f
	.string	"ptr"
	.byte	0x54
	.value	0x774
	.long	.LASF1984
	.long	0xab75
	.byte	0x1
	.long	0xd2e1
	.long	0xd2ec
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"ptr"
	.byte	0x54
	.value	0x775
	.long	.LASF1985
	.long	0x14088
	.byte	0x1
	.long	0xd305
	.long	0xd310
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"ptr"
	.byte	0x54
	.value	0x778
	.long	.LASF1986
	.long	0xab75
	.byte	0x1
	.long	0xd329
	.long	0xd339
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"ptr"
	.byte	0x54
	.value	0x779
	.long	.LASF1987
	.long	0x14088
	.byte	0x1
	.long	0xd352
	.long	0xd362
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"ptr"
	.byte	0x54
	.value	0x77c
	.long	.LASF1988
	.long	0xab75
	.byte	0x1
	.long	0xd37b
	.long	0xd390
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"ptr"
	.byte	0x54
	.value	0x77d
	.long	.LASF1989
	.long	0x14088
	.byte	0x1
	.long	0xd3a9
	.long	0xd3be
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"ptr"
	.byte	0x54
	.value	0x780
	.long	.LASF1990
	.long	0xab75
	.byte	0x1
	.long	0xd3d7
	.long	0xd3e2
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x9720
	.byte	0
	.uleb128 0x1f
	.string	"ptr"
	.byte	0x54
	.value	0x782
	.long	.LASF1991
	.long	0x14088
	.byte	0x1
	.long	0xd3fb
	.long	0xd406
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x9720
	.byte	0
	.uleb128 0x1c
	.long	.LASF1992
	.byte	0x54
	.value	0x7ee
	.long	.LASF1993
	.byte	0x2
	.long	0xd41b
	.long	0xd421
	.uleb128 0xb
	.long	0x14013
	.byte	0
	.uleb128 0x1e
	.long	.LASF1994
	.byte	0x2
	.value	0x1ab
	.long	.LASF1995
	.long	0xab81
	.byte	0x1
	.long	0xd443
	.long	0xd44e
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1996
	.byte	0x2
	.value	0x1b1
	.long	.LASF1997
	.long	0x1412e
	.byte	0x1
	.long	0xd470
	.long	0xd47b
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x912d
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1996
	.byte	0x2
	.value	0x1ab
	.long	.LASF1998
	.long	0x14128
	.byte	0x1
	.long	0xd49d
	.long	0xd4a8
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x912d
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1999
	.byte	0x2
	.value	0x1ab
	.long	.LASF2000
	.long	0x148cf
	.byte	0x1
	.long	0xd4ca
	.long	0xd4d5
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x12f49
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1999
	.byte	0x2
	.value	0x1b1
	.long	.LASF2001
	.long	0x148d5
	.byte	0x1
	.long	0xd4f7
	.long	0xd502
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x12f49
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF2002
	.byte	0x2
	.value	0x1ab
	.long	.LASF2003
	.long	0x143b0
	.byte	0x1
	.long	0xd524
	.long	0xd52f
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x117b2
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF2004
	.byte	0x2
	.value	0x1ab
	.long	.LASF2005
	.long	0x1490f
	.byte	0x1
	.long	0xd551
	.long	0xd55c
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x13a1c
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF2004
	.byte	0x2
	.value	0x1b1
	.long	.LASF2006
	.long	0x14915
	.byte	0x1
	.long	0xd57e
	.long	0xd589
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x13a1c
	.uleb128 0xb
	.long	0x1406a
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF2007
	.byte	0x2
	.value	0x1ab
	.long	.LASF2008
	.long	0x14320
	.byte	0x1
	.long	0xd5ab
	.long	0xd5b6
	.uleb128 0x2c
	.string	"_Tp"
	.long	0xb4af
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x55
	.long	.LASF2009
	.byte	0x2
	.value	0x1ab
	.long	.LASF2010
	.long	0xaa90
	.byte	0x1
	.long	0xd5d4
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x9c79
	.uleb128 0xb
	.long	0x14013
	.uleb128 0xc
	.long	0x30
	.byte	0
	.byte	0
	.uleb128 0x37
	.long	.LASF2011
	.byte	0x54
	.value	0x376
	.long	0xd5ec
	.uleb128 0x37
	.long	.LASF2012
	.byte	0x54
	.value	0x374
	.long	0xc031
	.uleb128 0x8d
	.long	.LASF2013
	.byte	0x8
	.byte	0x54
	.value	0x5d0
	.long	0xd5f8
	.long	0xd6ce
	.uleb128 0x8e
	.long	.LASF2014
	.long	0x14985
	.byte	0
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF2013
	.byte	0x54
	.value	0x5d3
	.long	.LASF2015
	.byte	0x1
	.long	0xd62b
	.long	0xd631
	.uleb128 0xb
	.long	0x1400d
	.byte	0
	.uleb128 0x8f
	.long	.LASF2017
	.byte	0x54
	.value	0x5d4
	.long	.LASF2018
	.byte	0x1
	.long	0xd5f8
	.byte	0x1
	.long	0xd64c
	.long	0xd657
	.uleb128 0xb
	.long	0x1400d
	.uleb128 0xb
	.long	0x30
	.byte	0
	.uleb128 0x90
	.long	.LASF335
	.byte	0x54
	.value	0x5d5
	.long	.LASF3076
	.byte	0x1
	.uleb128 0x2
	.byte	0x10
	.uleb128 0x2
	.long	0xd5f8
	.byte	0x1
	.long	0xd675
	.long	0xd69e
	.uleb128 0xb
	.long	0x1400d
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x9720
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x14957
	.uleb128 0xc
	.long	0x1499c
	.uleb128 0xc
	.long	0x1499c
	.uleb128 0xc
	.long	0x13fce
	.byte	0
	.uleb128 0x91
	.long	.LASF338
	.byte	0x54
	.value	0x5d7
	.long	.LASF2019
	.byte	0x1
	.uleb128 0x2
	.byte	0x10
	.uleb128 0x3
	.long	0xd5f8
	.byte	0x1
	.long	0xd6b8
	.uleb128 0xb
	.long	0x1400d
	.uleb128 0xc
	.long	0xaa8a
	.uleb128 0xc
	.long	0xab75
	.uleb128 0xc
	.long	0xab75
	.byte	0
	.byte	0
	.uleb128 0x37
	.long	.LASF2020
	.byte	0x54
	.value	0x3be
	.long	0xd6da
	.uleb128 0x54
	.long	.LASF2021
	.byte	0x20
	.byte	0x54
	.value	0x3a3
	.long	0xd824
	.uleb128 0x34
	.long	0xe239
	.byte	0
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF2022
	.byte	0x54
	.value	0x3a7
	.long	.LASF2023
	.byte	0x1
	.long	0xd703
	.long	0xd709
	.uleb128 0xb
	.long	0x140c3
	.byte	0
	.uleb128 0x1c
	.long	.LASF2022
	.byte	0x54
	.value	0x3a8
	.long	.LASF2024
	.byte	0x1
	.long	0xd71e
	.long	0xd738
	.uleb128 0xb
	.long	0x140c3
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2022
	.byte	0x54
	.value	0x3a9
	.long	.LASF2025
	.byte	0x1
	.long	0xd74d
	.long	0xd758
	.uleb128 0xb
	.long	0x140c3
	.uleb128 0xc
	.long	0x140c9
	.byte	0
	.uleb128 0x1c
	.long	.LASF2022
	.byte	0x54
	.value	0x3aa
	.long	.LASF2026
	.byte	0x1
	.long	0xd76d
	.long	0xd778
	.uleb128 0xb
	.long	0x140c3
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x86
	.string	"all"
	.byte	0x54
	.value	0x3ad
	.long	.LASF2027
	.long	0xd6da
	.byte	0x1
	.long	0xd794
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1e
	.long	.LASF1745
	.byte	0x54
	.value	0x3af
	.long	.LASF2028
	.long	0xade8
	.byte	0x1
	.long	0xd7ad
	.long	0xd7b3
	.uleb128 0xb
	.long	0x140d4
	.byte	0
	.uleb128 0x1f
	.string	"mul"
	.byte	0x54
	.value	0x3b5
	.long	.LASF2029
	.long	0xd6da
	.byte	0x1
	.long	0xd7cc
	.long	0xd7dc
	.uleb128 0xb
	.long	0x140d4
	.uleb128 0xc
	.long	0x140da
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1e
	.long	.LASF1741
	.byte	0x54
	.value	0x3b8
	.long	.LASF2030
	.long	0xd6da
	.byte	0x1
	.long	0xd7f5
	.long	0xd7fb
	.uleb128 0xb
	.long	0x140d4
	.byte	0
	.uleb128 0x1e
	.long	.LASF2031
	.byte	0x54
	.value	0x3bb
	.long	.LASF2032
	.long	0x9ef1
	.byte	0x1
	.long	0xd814
	.long	0xd81a
	.uleb128 0xb
	.long	0x140d4
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.byte	0
	.uleb128 0x14
	.long	0xd6ce
	.uleb128 0x14
	.long	0xc3f6
	.uleb128 0x14
	.long	0xb8ec
	.uleb128 0x4
	.string	"gpu"
	.byte	0x54
	.byte	0x67
	.long	0xd849
	.uleb128 0x2a
	.long	.LASF2033
	.uleb128 0x14
	.long	0xd83e
	.byte	0
	.uleb128 0x49
	.long	.LASF2034
	.value	0x160
	.byte	0x2
	.value	0x4c3
	.long	0xdb52
	.uleb128 0x87
	.string	"op"
	.byte	0x2
	.value	0x4ea
	.long	0x143d9
	.byte	0
	.byte	0x1
	.uleb128 0x51
	.long	.LASF1676
	.byte	0x2
	.value	0x4eb
	.long	0x30
	.byte	0x8
	.byte	0x1
	.uleb128 0x87
	.string	"a"
	.byte	0x2
	.value	0x4ed
	.long	0xc3f6
	.byte	0x10
	.byte	0x1
	.uleb128 0x87
	.string	"b"
	.byte	0x2
	.value	0x4ed
	.long	0xc3f6
	.byte	0x70
	.byte	0x1
	.uleb128 0x87
	.string	"c"
	.byte	0x2
	.value	0x4ed
	.long	0xc3f6
	.byte	0xd0
	.byte	0x1
	.uleb128 0x92
	.long	.LASF2035
	.byte	0x2
	.value	0x4ee
	.long	0x29
	.value	0x130
	.byte	0x1
	.uleb128 0x92
	.long	.LASF2036
	.byte	0x2
	.value	0x4ee
	.long	0x29
	.value	0x138
	.byte	0x1
	.uleb128 0x93
	.string	"s"
	.byte	0x2
	.value	0x4ef
	.long	0xd6ce
	.value	0x140
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF2034
	.byte	0x2
	.value	0x4c6
	.long	.LASF2037
	.byte	0x1
	.long	0xd8dd
	.long	0xd8e3
	.uleb128 0xb
	.long	0x143df
	.byte	0
	.uleb128 0x1c
	.long	.LASF2034
	.byte	0x2
	.value	0x4c7
	.long	.LASF2038
	.byte	0x1
	.long	0xd8f8
	.long	0xd926
	.uleb128 0xb
	.long	0x143df
	.uleb128 0xc
	.long	0x143d9
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x1401f
	.uleb128 0xc
	.long	0x1401f
	.uleb128 0xc
	.long	0x1401f
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x14019
	.byte	0
	.uleb128 0x1d
	.long	.LASF2034
	.byte	0x2
	.value	0x4ca
	.long	.LASF2039
	.byte	0x1
	.long	0xd93b
	.long	0xd946
	.uleb128 0xb
	.long	0x143df
	.uleb128 0xc
	.long	0x1401f
	.byte	0
	.uleb128 0x1e
	.long	.LASF2040
	.byte	0x2
	.value	0x4cb
	.long	.LASF2041
	.long	0xc3f6
	.byte	0x1
	.long	0xd95f
	.long	0xd965
	.uleb128 0xb
	.long	0x143e5
	.byte	0
	.uleb128 0x1f
	.string	"row"
	.byte	0x2
	.value	0x4d9
	.long	.LASF2042
	.long	0xd849
	.byte	0x1
	.long	0xd97e
	.long	0xd989
	.uleb128 0xb
	.long	0x143e5
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"col"
	.byte	0x2
	.value	0x4da
	.long	.LASF2043
	.long	0xd849
	.byte	0x1
	.long	0xd9a2
	.long	0xd9ad
	.uleb128 0xb
	.long	0x143e5
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1904
	.byte	0x2
	.value	0x4db
	.long	.LASF2044
	.long	0xd849
	.byte	0x1
	.long	0xd9c6
	.long	0xd9d1
	.uleb128 0xb
	.long	0x143e5
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x2
	.value	0x4dc
	.long	.LASF2045
	.long	0xd849
	.byte	0x1
	.long	0xd9ea
	.long	0xd9fa
	.uleb128 0xb
	.long	0x143e5
	.uleb128 0xc
	.long	0x1402b
	.uleb128 0xc
	.long	0x1402b
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x2
	.value	0x4dd
	.long	.LASF2046
	.long	0xd849
	.byte	0x1
	.long	0xda13
	.long	0xda1e
	.uleb128 0xb
	.long	0x143e5
	.uleb128 0xc
	.long	0x14031
	.byte	0
	.uleb128 0x1e
	.long	.LASF1743
	.byte	0x2
	.value	0x4df
	.long	.LASF2047
	.long	0xc3f6
	.byte	0x1
	.long	0xda37
	.long	0xda42
	.uleb128 0xb
	.long	0x143e5
	.uleb128 0xc
	.long	0x1401f
	.byte	0
	.uleb128 0x1f
	.string	"dot"
	.byte	0x2
	.value	0x4e0
	.long	.LASF2048
	.long	0x29
	.byte	0x1
	.long	0xda5b
	.long	0xda66
	.uleb128 0xb
	.long	0x143e5
	.uleb128 0xc
	.long	0x1401f
	.byte	0
	.uleb128 0x1f
	.string	"t"
	.byte	0x2
	.value	0x4e2
	.long	.LASF2049
	.long	0xd849
	.byte	0x1
	.long	0xda7d
	.long	0xda83
	.uleb128 0xb
	.long	0x143e5
	.byte	0
	.uleb128 0x1f
	.string	"inv"
	.byte	0x2
	.value	0x4e3
	.long	.LASF2050
	.long	0xd849
	.byte	0x1
	.long	0xda9c
	.long	0xdaa7
	.uleb128 0xb
	.long	0x143e5
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"mul"
	.byte	0x2
	.value	0x4e4
	.long	.LASF2051
	.long	0xd849
	.byte	0x1
	.long	0xdac0
	.long	0xdad0
	.uleb128 0xb
	.long	0x143e5
	.uleb128 0xc
	.long	0x14064
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1f
	.string	"mul"
	.byte	0x2
	.value	0x4e5
	.long	.LASF2052
	.long	0xd849
	.byte	0x1
	.long	0xdae9
	.long	0xdaf9
	.uleb128 0xb
	.long	0x143e5
	.uleb128 0xc
	.long	0x1401f
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1e
	.long	.LASF115
	.byte	0x2
	.value	0x4e7
	.long	.LASF2053
	.long	0xd5e0
	.byte	0x1
	.long	0xdb12
	.long	0xdb18
	.uleb128 0xb
	.long	0x143e5
	.byte	0
	.uleb128 0x1e
	.long	.LASF1649
	.byte	0x2
	.value	0x4e8
	.long	.LASF2054
	.long	0x30
	.byte	0x1
	.long	0xdb31
	.long	0xdb37
	.uleb128 0xb
	.long	0x143e5
	.byte	0
	.uleb128 0x94
	.long	.LASF2055
	.long	.LASF3077
	.byte	0x1
	.long	0xdb46
	.uleb128 0xb
	.long	0x143df
	.uleb128 0xb
	.long	0x30
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0xd849
	.uleb128 0x37
	.long	.LASF2056
	.byte	0x54
	.value	0x5bf
	.long	0x14070
	.uleb128 0x2a
	.long	.LASF2057
	.uleb128 0x14
	.long	0xdb63
	.uleb128 0x37
	.long	.LASF2058
	.byte	0x54
	.value	0x5bd
	.long	0x14076
	.uleb128 0x14
	.long	0xc02c
	.uleb128 0x37
	.long	.LASF2059
	.byte	0x54
	.value	0x373
	.long	0xdb8a
	.uleb128 0x37
	.long	.LASF2060
	.byte	0x54
	.value	0x372
	.long	0xc1aa
	.uleb128 0x54
	.long	.LASF2061
	.byte	0x20
	.byte	0x54
	.value	0x1c3
	.long	0xe225
	.uleb128 0x87
	.string	"val"
	.byte	0x54
	.value	0x21e
	.long	0xadd8
	.byte	0
	.byte	0x1
	.uleb128 0x42
	.long	.LASF2062
	.byte	0x54
	.value	0x1c7
	.long	0xe225
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1cd
	.long	.LASF2064
	.byte	0x1
	.long	0xdbd4
	.long	0xdbda
	.uleb128 0xb
	.long	0x14093
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1cf
	.long	.LASF2065
	.byte	0x1
	.long	0xdbef
	.long	0xdbfa
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d0
	.long	.LASF2066
	.byte	0x1
	.long	0xdc0f
	.long	0xdc1f
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d1
	.long	.LASF2067
	.byte	0x1
	.long	0xdc34
	.long	0xdc49
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d2
	.long	.LASF2068
	.byte	0x1
	.long	0xdc5e
	.long	0xdc78
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d3
	.long	.LASF2069
	.byte	0x1
	.long	0xdc8d
	.long	0xdcac
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d4
	.long	.LASF2070
	.byte	0x1
	.long	0xdcc1
	.long	0xdce5
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d5
	.long	.LASF2071
	.byte	0x1
	.long	0xdcfa
	.long	0xdd23
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d6
	.long	.LASF2072
	.byte	0x1
	.long	0xdd38
	.long	0xdd66
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d7
	.long	.LASF2073
	.byte	0x1
	.long	0xdd7b
	.long	0xddae
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d8
	.long	.LASF2074
	.byte	0x1
	.long	0xddc3
	.long	0xddfb
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d9
	.long	.LASF2075
	.byte	0x1
	.long	0xde10
	.long	0xde52
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1dc
	.long	.LASF2076
	.byte	0x1
	.long	0xde67
	.long	0xdebd
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1d
	.long	.LASF2063
	.byte	0x54
	.value	0x1e0
	.long	.LASF2077
	.byte	0x1
	.long	0xded2
	.long	0xdedd
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0xb2ff
	.byte	0
	.uleb128 0x86
	.string	"all"
	.byte	0x54
	.value	0x1e2
	.long	.LASF2078
	.long	0xdb96
	.byte	0x1
	.long	0xdef9
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x95
	.long	.LASF1926
	.byte	0x54
	.value	0x1e3
	.long	.LASF2079
	.long	0xdb96
	.byte	0x1
	.uleb128 0x95
	.long	.LASF1930
	.byte	0x54
	.value	0x1e4
	.long	.LASF2080
	.long	0xdb96
	.byte	0x1
	.uleb128 0x4d
	.string	"eye"
	.byte	0x54
	.value	0x1e5
	.long	.LASF2081
	.long	0xdb96
	.byte	0x1
	.uleb128 0x8c
	.long	.LASF1904
	.byte	0x54
	.value	0x1e6
	.long	.LASF2082
	.long	0xdb96
	.byte	0x1
	.long	0xdf4a
	.uleb128 0xc
	.long	0x14099
	.byte	0
	.uleb128 0x14
	.long	0xdbb2
	.uleb128 0x8c
	.long	.LASF2083
	.byte	0x54
	.value	0x1e7
	.long	.LASF2084
	.long	0xdb96
	.byte	0x1
	.long	0xdf70
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x8c
	.long	.LASF2085
	.byte	0x54
	.value	0x1e8
	.long	.LASF2086
	.long	0xdb96
	.byte	0x1
	.long	0xdf91
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1f
	.string	"dot"
	.byte	0x54
	.value	0x1eb
	.long	.LASF2087
	.long	0x29
	.byte	0x1
	.long	0xdfaa
	.long	0xdfb5
	.uleb128 0xb
	.long	0x1409f
	.uleb128 0xc
	.long	0x140a5
	.byte	0
	.uleb128 0x1e
	.long	.LASF1839
	.byte	0x54
	.value	0x1ee
	.long	.LASF2088
	.long	0x29
	.byte	0x1
	.long	0xdfce
	.long	0xdfd9
	.uleb128 0xb
	.long	0x1409f
	.uleb128 0xc
	.long	0x140a5
	.byte	0
	.uleb128 0x1f
	.string	"row"
	.byte	0x54
	.value	0x1fa
	.long	.LASF2089
	.long	0xe225
	.byte	0x1
	.long	0xdff2
	.long	0xdffd
	.uleb128 0xb
	.long	0x1409f
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"col"
	.byte	0x54
	.value	0x1fd
	.long	.LASF2090
	.long	0xdb96
	.byte	0x1
	.long	0xe016
	.long	0xe021
	.uleb128 0xb
	.long	0x1409f
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1904
	.byte	0x54
	.value	0x200
	.long	.LASF2091
	.long	0xdbb2
	.byte	0x1
	.long	0xe03a
	.long	0xe040
	.uleb128 0xb
	.long	0x1409f
	.byte	0
	.uleb128 0x1f
	.string	"t"
	.byte	0x54
	.value	0x203
	.long	.LASF2092
	.long	0xe22f
	.byte	0x1
	.long	0xe057
	.long	0xe05d
	.uleb128 0xb
	.long	0x1409f
	.byte	0
	.uleb128 0x1f
	.string	"inv"
	.byte	0x54
	.value	0x206
	.long	.LASF2093
	.long	0xe22f
	.byte	0x1
	.long	0xe076
	.long	0xe081
	.uleb128 0xb
	.long	0x1409f
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF2094
	.byte	0x54
	.value	0x20a
	.long	.LASF2095
	.long	0xe234
	.byte	0x1
	.long	0xe09a
	.long	0xe0aa
	.uleb128 0xb
	.long	0x1409f
	.uleb128 0xc
	.long	0x140ab
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"mul"
	.byte	0x54
	.value	0x20d
	.long	.LASF2096
	.long	0xdb96
	.byte	0x1
	.long	0xe0c3
	.long	0xe0ce
	.uleb128 0xb
	.long	0x1409f
	.uleb128 0xc
	.long	0x140a5
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x210
	.long	.LASF2097
	.long	0xb334
	.byte	0x1
	.long	0xe0e7
	.long	0xe0f7
	.uleb128 0xb
	.long	0x1409f
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x211
	.long	.LASF2098
	.long	0xb32e
	.byte	0x1
	.long	0xe110
	.long	0xe120
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x214
	.long	.LASF2099
	.long	0xb334
	.byte	0x1
	.long	0xe139
	.long	0xe144
	.uleb128 0xb
	.long	0x1409f
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x215
	.long	.LASF2100
	.long	0xb32e
	.byte	0x1
	.long	0xe15d
	.long	0xe168
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x217
	.long	.LASF2101
	.byte	0x1
	.long	0xe17d
	.long	0xe192
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x140a5
	.uleb128 0xc
	.long	0x140a5
	.uleb128 0xc
	.long	0xb48b
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x218
	.long	.LASF2102
	.byte	0x1
	.long	0xe1a7
	.long	0xe1bc
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x140a5
	.uleb128 0xc
	.long	0x140a5
	.uleb128 0xc
	.long	0xb494
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x21a
	.long	.LASF2103
	.byte	0x1
	.long	0xe1d1
	.long	0xe1e6
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x140a5
	.uleb128 0xc
	.long	0x140a5
	.uleb128 0xc
	.long	0xb49d
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x21c
	.long	.LASF2104
	.byte	0x1
	.long	0xe1fb
	.long	0xe20b
	.uleb128 0xb
	.long	0x14093
	.uleb128 0xc
	.long	0x140b1
	.uleb128 0xc
	.long	0xb4a6
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.uleb128 0x2e
	.string	"m"
	.long	0x30
	.byte	0x4
	.uleb128 0x2e
	.string	"n"
	.long	0x30
	.byte	0x1
	.byte	0
	.uleb128 0x2a
	.long	.LASF2105
	.uleb128 0x14
	.long	0xdb96
	.uleb128 0x2a
	.long	.LASF2106
	.uleb128 0x2a
	.long	.LASF2107
	.uleb128 0x54
	.long	.LASF2108
	.byte	0x20
	.byte	0x54
	.value	0x258
	.long	0xe671
	.uleb128 0x34
	.long	0xdb96
	.byte	0
	.byte	0x1
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x25f
	.long	.LASF2109
	.byte	0x1
	.long	0xe263
	.long	0xe269
	.uleb128 0xb
	.long	0x140b7
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x261
	.long	.LASF2110
	.byte	0x1
	.long	0xe27f
	.long	0xe28a
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x262
	.long	.LASF2111
	.byte	0x1
	.long	0xe2a0
	.long	0xe2b0
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x263
	.long	.LASF2112
	.byte	0x1
	.long	0xe2c6
	.long	0xe2db
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x264
	.long	.LASF2113
	.byte	0x1
	.long	0xe2f1
	.long	0xe30b
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x265
	.long	.LASF2114
	.byte	0x1
	.long	0xe321
	.long	0xe340
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x266
	.long	.LASF2115
	.byte	0x1
	.long	0xe356
	.long	0xe37a
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x267
	.long	.LASF2116
	.byte	0x1
	.long	0xe390
	.long	0xe3b9
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x268
	.long	.LASF2117
	.byte	0x1
	.long	0xe3cf
	.long	0xe3fd
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x269
	.long	.LASF2118
	.byte	0x1
	.long	0xe413
	.long	0xe446
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x26a
	.long	.LASF2119
	.byte	0x1
	.long	0xe45c
	.long	0xe494
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x85
	.string	"Vec"
	.byte	0x54
	.value	0x26b
	.long	.LASF2120
	.byte	0x1
	.long	0xe4aa
	.long	0xe4b5
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0xb2ff
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x26d
	.long	.LASF2121
	.byte	0x1
	.long	0xe4cb
	.long	0xe4d6
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x140ab
	.byte	0
	.uleb128 0x86
	.string	"all"
	.byte	0x54
	.value	0x26f
	.long	.LASF2122
	.long	0xe239
	.byte	0x1
	.long	0xe4f2
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1f
	.string	"mul"
	.byte	0x54
	.value	0x272
	.long	.LASF2123
	.long	0xe239
	.byte	0x1
	.long	0xe50b
	.long	0xe516
	.uleb128 0xb
	.long	0x140bd
	.uleb128 0xc
	.long	0x140ab
	.byte	0
	.uleb128 0x1e
	.long	.LASF1741
	.byte	0x54
	.value	0x275
	.long	.LASF2124
	.long	0xe239
	.byte	0x1
	.long	0xe52f
	.long	0xe535
	.uleb128 0xb
	.long	0x140bd
	.byte	0
	.uleb128 0x1e
	.long	.LASF1743
	.byte	0x54
	.value	0x27c
	.long	.LASF2125
	.long	0xe239
	.byte	0x1
	.long	0xe54e
	.long	0xe559
	.uleb128 0xb
	.long	0x140bd
	.uleb128 0xc
	.long	0x140ab
	.byte	0
	.uleb128 0x1e
	.long	.LASF1745
	.byte	0x54
	.value	0x280
	.long	.LASF2126
	.long	0xade8
	.byte	0x1
	.long	0xe572
	.long	0xe578
	.uleb128 0xb
	.long	0x140bd
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x54
	.value	0x283
	.long	.LASF2127
	.long	0xb334
	.byte	0x1
	.long	0xe591
	.long	0xe59c
	.uleb128 0xb
	.long	0x140bd
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x54
	.value	0x284
	.long	.LASF2128
	.long	0xb32e
	.byte	0x1
	.long	0xe5b5
	.long	0xe5c0
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x285
	.long	.LASF2129
	.long	0xb334
	.byte	0x1
	.long	0xe5d9
	.long	0xe5e4
	.uleb128 0xb
	.long	0x140bd
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x286
	.long	.LASF2130
	.long	0xb32e
	.byte	0x1
	.long	0xe5fd
	.long	0xe608
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x288
	.long	.LASF2131
	.byte	0x1
	.long	0xe61e
	.long	0xe633
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x140a5
	.uleb128 0xc
	.long	0x140a5
	.uleb128 0xc
	.long	0xb48b
	.byte	0
	.uleb128 0x84
	.string	"Vec"
	.byte	0x54
	.value	0x289
	.long	.LASF2132
	.byte	0x1
	.long	0xe649
	.long	0xe65e
	.uleb128 0xb
	.long	0x140b7
	.uleb128 0xc
	.long	0x140a5
	.uleb128 0xc
	.long	0x140a5
	.uleb128 0xc
	.long	0xb494
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.uleb128 0x2e
	.string	"cn"
	.long	0x30
	.byte	0x4
	.byte	0
	.uleb128 0x14
	.long	0xe239
	.uleb128 0x14
	.long	0xe22f
	.uleb128 0x14
	.long	0xd6da
	.uleb128 0x14
	.long	0xbbf4
	.uleb128 0x14
	.long	0xb8f8
	.uleb128 0x54
	.long	.LASF2133
	.byte	0x8
	.byte	0x54
	.value	0x1c3
	.long	0xed19
	.uleb128 0x87
	.string	"val"
	.byte	0x54
	.value	0x21e
	.long	0x9178
	.byte	0
	.byte	0x1
	.uleb128 0x42
	.long	.LASF2062
	.byte	0x54
	.value	0x1c7
	.long	0xed19
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1cd
	.long	.LASF2134
	.byte	0x1
	.long	0xe6c8
	.long	0xe6ce
	.uleb128 0xb
	.long	0x140fe
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1cf
	.long	.LASF2135
	.byte	0x1
	.long	0xe6e3
	.long	0xe6ee
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d0
	.long	.LASF2136
	.byte	0x1
	.long	0xe703
	.long	0xe713
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d1
	.long	.LASF2137
	.byte	0x1
	.long	0xe728
	.long	0xe73d
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d2
	.long	.LASF2138
	.byte	0x1
	.long	0xe752
	.long	0xe76c
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d3
	.long	.LASF2139
	.byte	0x1
	.long	0xe781
	.long	0xe7a0
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d4
	.long	.LASF2140
	.byte	0x1
	.long	0xe7b5
	.long	0xe7d9
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d5
	.long	.LASF2141
	.byte	0x1
	.long	0xe7ee
	.long	0xe817
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d6
	.long	.LASF2142
	.byte	0x1
	.long	0xe82c
	.long	0xe85a
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d7
	.long	.LASF2143
	.byte	0x1
	.long	0xe86f
	.long	0xe8a2
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d8
	.long	.LASF2144
	.byte	0x1
	.long	0xe8b7
	.long	0xe8ef
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1d9
	.long	.LASF2145
	.byte	0x1
	.long	0xe904
	.long	0xe946
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x1dc
	.long	.LASF2146
	.byte	0x1
	.long	0xe95b
	.long	0xe9b1
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1d
	.long	.LASF2063
	.byte	0x54
	.value	0x1e0
	.long	.LASF2147
	.byte	0x1
	.long	0xe9c6
	.long	0xe9d1
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x9720
	.byte	0
	.uleb128 0x86
	.string	"all"
	.byte	0x54
	.value	0x1e2
	.long	.LASF2148
	.long	0xe68a
	.byte	0x1
	.long	0xe9ed
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x95
	.long	.LASF1926
	.byte	0x54
	.value	0x1e3
	.long	.LASF2149
	.long	0xe68a
	.byte	0x1
	.uleb128 0x95
	.long	.LASF1930
	.byte	0x54
	.value	0x1e4
	.long	.LASF2150
	.long	0xe68a
	.byte	0x1
	.uleb128 0x4d
	.string	"eye"
	.byte	0x54
	.value	0x1e5
	.long	.LASF2151
	.long	0xe68a
	.byte	0x1
	.uleb128 0x8c
	.long	.LASF1904
	.byte	0x54
	.value	0x1e6
	.long	.LASF2152
	.long	0xe68a
	.byte	0x1
	.long	0xea3e
	.uleb128 0xc
	.long	0x14104
	.byte	0
	.uleb128 0x14
	.long	0xe6a6
	.uleb128 0x8c
	.long	.LASF2083
	.byte	0x54
	.value	0x1e7
	.long	.LASF2153
	.long	0xe68a
	.byte	0x1
	.long	0xea64
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x8c
	.long	.LASF2085
	.byte	0x54
	.value	0x1e8
	.long	.LASF2154
	.long	0xe68a
	.byte	0x1
	.long	0xea85
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"dot"
	.byte	0x54
	.value	0x1eb
	.long	.LASF2155
	.long	0x30
	.byte	0x1
	.long	0xea9e
	.long	0xeaa9
	.uleb128 0xb
	.long	0x1410a
	.uleb128 0xc
	.long	0x14110
	.byte	0
	.uleb128 0x1e
	.long	.LASF1839
	.byte	0x54
	.value	0x1ee
	.long	.LASF2156
	.long	0x29
	.byte	0x1
	.long	0xeac2
	.long	0xeacd
	.uleb128 0xb
	.long	0x1410a
	.uleb128 0xc
	.long	0x14110
	.byte	0
	.uleb128 0x1f
	.string	"row"
	.byte	0x54
	.value	0x1fa
	.long	.LASF2157
	.long	0xed19
	.byte	0x1
	.long	0xeae6
	.long	0xeaf1
	.uleb128 0xb
	.long	0x1410a
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"col"
	.byte	0x54
	.value	0x1fd
	.long	.LASF2158
	.long	0xe68a
	.byte	0x1
	.long	0xeb0a
	.long	0xeb15
	.uleb128 0xb
	.long	0x1410a
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1904
	.byte	0x54
	.value	0x200
	.long	.LASF2159
	.long	0xe6a6
	.byte	0x1
	.long	0xeb2e
	.long	0xeb34
	.uleb128 0xb
	.long	0x1410a
	.byte	0
	.uleb128 0x1f
	.string	"t"
	.byte	0x54
	.value	0x203
	.long	.LASF2160
	.long	0xed23
	.byte	0x1
	.long	0xeb4b
	.long	0xeb51
	.uleb128 0xb
	.long	0x1410a
	.byte	0
	.uleb128 0x1f
	.string	"inv"
	.byte	0x54
	.value	0x206
	.long	.LASF2161
	.long	0xed23
	.byte	0x1
	.long	0xeb6a
	.long	0xeb75
	.uleb128 0xb
	.long	0x1410a
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF2094
	.byte	0x54
	.value	0x20a
	.long	.LASF2162
	.long	0xed28
	.byte	0x1
	.long	0xeb8e
	.long	0xeb9e
	.uleb128 0xb
	.long	0x1410a
	.uleb128 0xc
	.long	0x140e6
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"mul"
	.byte	0x54
	.value	0x20d
	.long	.LASF2163
	.long	0xe68a
	.byte	0x1
	.long	0xebb7
	.long	0xebc2
	.uleb128 0xb
	.long	0x1410a
	.uleb128 0xc
	.long	0x14110
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x210
	.long	.LASF2164
	.long	0x13fbc
	.byte	0x1
	.long	0xebdb
	.long	0xebeb
	.uleb128 0xb
	.long	0x1410a
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x211
	.long	.LASF2165
	.long	0x13fc2
	.byte	0x1
	.long	0xec04
	.long	0xec14
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x214
	.long	.LASF2166
	.long	0x13fbc
	.byte	0x1
	.long	0xec2d
	.long	0xec38
	.uleb128 0xb
	.long	0x1410a
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0x54
	.value	0x215
	.long	.LASF2167
	.long	0x13fc2
	.byte	0x1
	.long	0xec51
	.long	0xec5c
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x217
	.long	.LASF2168
	.byte	0x1
	.long	0xec71
	.long	0xec86
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x14110
	.uleb128 0xc
	.long	0x14110
	.uleb128 0xc
	.long	0xb48b
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x218
	.long	.LASF2169
	.byte	0x1
	.long	0xec9b
	.long	0xecb0
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x14110
	.uleb128 0xc
	.long	0x14110
	.uleb128 0xc
	.long	0xb494
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x21a
	.long	.LASF2170
	.byte	0x1
	.long	0xecc5
	.long	0xecda
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x14110
	.uleb128 0xc
	.long	0x14110
	.uleb128 0xc
	.long	0xb49d
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0x54
	.value	0x21c
	.long	.LASF2171
	.byte	0x1
	.long	0xecef
	.long	0xecff
	.uleb128 0xb
	.long	0x140fe
	.uleb128 0xc
	.long	0x14116
	.uleb128 0xc
	.long	0xb4a6
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x30
	.uleb128 0x2e
	.string	"m"
	.long	0x30
	.byte	0x2
	.uleb128 0x2e
	.string	"n"
	.long	0x30
	.byte	0x1
	.byte	0
	.uleb128 0x2a
	.long	.LASF2172
	.uleb128 0x14
	.long	0xe68a
	.uleb128 0x2a
	.long	.LASF2173
	.uleb128 0x2a
	.long	.LASF2174
	.uleb128 0x14
	.long	0xed23
	.uleb128 0x2a
	.long	.LASF2175
	.uleb128 0x54
	.long	.LASF2176
	.byte	0x10
	.byte	0x54
	.value	0x1059
	.long	0xf0bf
	.uleb128 0x87
	.string	"fs"
	.byte	0x54
	.value	0x10aa
	.long	0x1429a
	.byte	0
	.byte	0x1
	.uleb128 0x51
	.long	.LASF2177
	.byte	0x54
	.value	0x10ab
	.long	0x142a5
	.byte	0x8
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF2176
	.byte	0x54
	.value	0x106f
	.long	.LASF2178
	.byte	0x1
	.long	0xed75
	.long	0xed7b
	.uleb128 0xb
	.long	0x142b0
	.byte	0
	.uleb128 0x1c
	.long	.LASF2176
	.byte	0x54
	.value	0x1071
	.long	.LASF2179
	.byte	0x1
	.long	0xed90
	.long	0xeda0
	.uleb128 0xb
	.long	0x142b0
	.uleb128 0xc
	.long	0x1429a
	.uleb128 0xc
	.long	0x142a5
	.byte	0
	.uleb128 0x1c
	.long	.LASF2176
	.byte	0x54
	.value	0x1073
	.long	.LASF2180
	.byte	0x1
	.long	0xedb5
	.long	0xedc0
	.uleb128 0xb
	.long	0x142b0
	.uleb128 0xc
	.long	0x142b6
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x54
	.value	0x1075
	.long	.LASF2181
	.long	0xed37
	.byte	0x1
	.long	0xedd9
	.long	0xede4
	.uleb128 0xb
	.long	0x142bc
	.uleb128 0xc
	.long	0xa63b
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x54
	.value	0x1077
	.long	.LASF2182
	.long	0xed37
	.byte	0x1
	.long	0xedfd
	.long	0xee08
	.uleb128 0xb
	.long	0x142bc
	.uleb128 0xc
	.long	0x9472
	.byte	0
	.uleb128 0x1e
	.long	.LASF134
	.byte	0x54
	.value	0x1079
	.long	.LASF2183
	.long	0xed37
	.byte	0x1
	.long	0xee21
	.long	0xee2c
	.uleb128 0xb
	.long	0x142bc
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1649
	.byte	0x54
	.value	0x107b
	.long	.LASF2184
	.long	0x30
	.byte	0x1
	.long	0xee45
	.long	0xee4b
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF132
	.byte	0x54
	.value	0x107e
	.long	.LASF2185
	.long	0x9ef1
	.byte	0x1
	.long	0xee64
	.long	0xee6a
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF2186
	.byte	0x54
	.value	0x1080
	.long	.LASF2187
	.long	0x9ef1
	.byte	0x1
	.long	0xee83
	.long	0xee89
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF2188
	.byte	0x54
	.value	0x1082
	.long	.LASF2189
	.long	0x9ef1
	.byte	0x1
	.long	0xeea2
	.long	0xeea8
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF2190
	.byte	0x54
	.value	0x1084
	.long	.LASF2191
	.long	0x9ef1
	.byte	0x1
	.long	0xeec1
	.long	0xeec7
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF2192
	.byte	0x54
	.value	0x1086
	.long	.LASF2193
	.long	0x9ef1
	.byte	0x1
	.long	0xeee0
	.long	0xeee6
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF2031
	.byte	0x54
	.value	0x1088
	.long	.LASF2194
	.long	0x9ef1
	.byte	0x1
	.long	0xeeff
	.long	0xef05
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF2195
	.byte	0x54
	.value	0x108a
	.long	.LASF2196
	.long	0x9ef1
	.byte	0x1
	.long	0xef1e
	.long	0xef24
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF2197
	.byte	0x54
	.value	0x108c
	.long	.LASF2198
	.long	0x9ef1
	.byte	0x1
	.long	0xef3d
	.long	0xef43
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF2199
	.byte	0x54
	.value	0x108e
	.long	.LASF2200
	.long	0x1a25
	.byte	0x1
	.long	0xef5c
	.long	0xef62
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF115
	.byte	0x54
	.value	0x1090
	.long	.LASF2201
	.long	0x911b
	.byte	0x1
	.long	0xef7b
	.long	0xef81
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF2202
	.byte	0x54
	.value	0x1092
	.long	.LASF2203
	.long	0x30
	.byte	0x1
	.long	0xef9a
	.long	0xefa0
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF2204
	.byte	0x54
	.value	0x1094
	.long	.LASF2205
	.long	0x9c79
	.byte	0x1
	.long	0xefb9
	.long	0xefbf
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF2206
	.byte	0x54
	.value	0x1096
	.long	.LASF2207
	.long	0x29
	.byte	0x1
	.long	0xefd8
	.long	0xefde
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF2208
	.byte	0x54
	.value	0x1098
	.long	.LASF2209
	.long	0x1a25
	.byte	0x1
	.long	0xeff7
	.long	0xeffd
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF1169
	.byte	0x54
	.value	0x109b
	.long	.LASF2210
	.long	0xb28d
	.byte	0x1
	.long	0xf016
	.long	0xf01c
	.uleb128 0xb
	.long	0x142b0
	.byte	0
	.uleb128 0x1e
	.long	.LASF1169
	.byte	0x54
	.value	0x109d
	.long	.LASF2211
	.long	0x142a5
	.byte	0x1
	.long	0xf035
	.long	0xf03b
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1e
	.long	.LASF96
	.byte	0x54
	.value	0x10a0
	.long	.LASF2212
	.long	0xf0c4
	.byte	0x1
	.long	0xf054
	.long	0xf05a
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1f
	.string	"end"
	.byte	0x54
	.value	0x10a2
	.long	.LASF2213
	.long	0xf0c4
	.byte	0x1
	.long	0xf073
	.long	0xf079
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.uleb128 0x1c
	.long	.LASF2214
	.byte	0x54
	.value	0x10a5
	.long	.LASF2215
	.byte	0x1
	.long	0xf08e
	.long	0xf0a3
	.uleb128 0xb
	.long	0x142bc
	.uleb128 0xc
	.long	0xa63b
	.uleb128 0xc
	.long	0xab75
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x55
	.long	.LASF2216
	.byte	0x54
	.value	0x10a7
	.long	.LASF2217
	.long	0x919a
	.byte	0x1
	.long	0xf0b8
	.uleb128 0xb
	.long	0x142bc
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0xed37
	.uleb128 0x54
	.long	.LASF2218
	.byte	0x58
	.byte	0x54
	.value	0x10b4
	.long	0xf2a5
	.uleb128 0x87
	.string	"fs"
	.byte	0x54
	.value	0x10d3
	.long	0x1429a
	.byte	0
	.byte	0x1
	.uleb128 0x51
	.long	.LASF2219
	.byte	0x54
	.value	0x10d4
	.long	0x142a5
	.byte	0x8
	.byte	0x1
	.uleb128 0x51
	.long	.LASF2220
	.byte	0x54
	.value	0x10d5
	.long	0xb067
	.byte	0x10
	.byte	0x1
	.uleb128 0x51
	.long	.LASF2221
	.byte	0x54
	.value	0x10d6
	.long	0x911b
	.byte	0x50
	.byte	0x1
	.uleb128 0x1c
	.long	.LASF2218
	.byte	0x54
	.value	0x10b8
	.long	.LASF2222
	.byte	0x1
	.long	0xf11e
	.long	0xf124
	.uleb128 0xb
	.long	0x142c2
	.byte	0
	.uleb128 0x1c
	.long	.LASF2218
	.byte	0x54
	.value	0x10ba
	.long	.LASF2223
	.byte	0x1
	.long	0xf139
	.long	0xf14e
	.uleb128 0xb
	.long	0x142c2
	.uleb128 0xc
	.long	0x1429a
	.uleb128 0xc
	.long	0x142a5
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.uleb128 0x1c
	.long	.LASF2218
	.byte	0x54
	.value	0x10bc
	.long	.LASF2224
	.byte	0x1
	.long	0xf163
	.long	0xf16e
	.uleb128 0xb
	.long	0x142c2
	.uleb128 0xc
	.long	0x142c8
	.byte	0
	.uleb128 0x1e
	.long	.LASF1169
	.byte	0x54
	.value	0x10be
	.long	.LASF2225
	.long	0xed37
	.byte	0x1
	.long	0xf187
	.long	0xf18d
	.uleb128 0xb
	.long	0x142ce
	.byte	0
	.uleb128 0x1e
	.long	.LASF1171
	.byte	0x54
	.value	0x10c0
	.long	.LASF2226
	.long	0xed37
	.byte	0x1
	.long	0xf1a6
	.long	0xf1ac
	.uleb128 0xb
	.long	0x142ce
	.byte	0
	.uleb128 0x1e
	.long	.LASF1173
	.byte	0x54
	.value	0x10c3
	.long	.LASF2227
	.long	0x142d4
	.byte	0x1
	.long	0xf1c5
	.long	0xf1cb
	.uleb128 0xb
	.long	0x142c2
	.byte	0
	.uleb128 0x1e
	.long	.LASF1173
	.byte	0x54
	.value	0x10c5
	.long	.LASF2228
	.long	0xf0c4
	.byte	0x1
	.long	0xf1e4
	.long	0xf1ef
	.uleb128 0xb
	.long	0x142c2
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1176
	.byte	0x54
	.value	0x10c7
	.long	.LASF2229
	.long	0x142d4
	.byte	0x1
	.long	0xf208
	.long	0xf20e
	.uleb128 0xb
	.long	0x142c2
	.byte	0
	.uleb128 0x1e
	.long	.LASF1176
	.byte	0x54
	.value	0x10c9
	.long	.LASF2230
	.long	0xf0c4
	.byte	0x1
	.long	0xf227
	.long	0xf232
	.uleb128 0xb
	.long	0x142c2
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF145
	.byte	0x54
	.value	0x10cb
	.long	.LASF2231
	.long	0x142d4
	.byte	0x1
	.long	0xf24b
	.long	0xf256
	.uleb128 0xb
	.long	0x142c2
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF637
	.byte	0x54
	.value	0x10cd
	.long	.LASF2232
	.long	0x142d4
	.byte	0x1
	.long	0xf26f
	.long	0xf27a
	.uleb128 0xb
	.long	0x142c2
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x55
	.long	.LASF2214
	.byte	0x54
	.value	0x10d0
	.long	.LASF2233
	.long	0x142d4
	.byte	0x1
	.long	0xf28f
	.uleb128 0xb
	.long	0x142c2
	.uleb128 0xc
	.long	0xa63b
	.uleb128 0xc
	.long	0xab75
	.uleb128 0xc
	.long	0x911b
	.byte	0
	.byte	0
	.uleb128 0x14
	.long	0xf0c4
	.uleb128 0x54
	.long	.LASF2234
	.byte	0x8
	.byte	0x54
	.value	0x1c3
	.long	0xf931
	.uleb128 0x87
	.string	"val"
	.byte	0x54
	.value	0x21e
	.long	0x142ec
	.byte	0
	.byte	0x1
	.uleb128 0x42
	.long	.LASF2062
	.byte	0x54
	.value	0x1c7
	.long	0xf931
	.byte	0x1
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xce
	.long	.LASF2235
	.byte	0x1
	.long	0xf2e7
	.long	0xf2ed
	.uleb128 0xb
	.long	0x142fc
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xd3
	.long	.LASF2236
	.byte	0x1
	.long	0xf301
	.long	0xf30c
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xd9
	.long	.LASF2237
	.byte	0x1
	.long	0xf320
	.long	0xf330
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xe0
	.long	.LASF2238
	.byte	0x1
	.long	0xf344
	.long	0xf359
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xe7
	.long	.LASF2239
	.byte	0x1
	.long	0xf36d
	.long	0xf387
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xee
	.long	.LASF2240
	.byte	0x1
	.long	0xf39b
	.long	0xf3ba
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xf5
	.long	.LASF2241
	.byte	0x1
	.long	0xf3ce
	.long	0xf3f2
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xfe
	.long	.LASF2242
	.byte	0x1
	.long	0xf406
	.long	0xf42f
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x107
	.long	.LASF2243
	.byte	0x1
	.long	0xf444
	.long	0xf472
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x110
	.long	.LASF2244
	.byte	0x1
	.long	0xf487
	.long	0xf4ba
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x11b
	.long	.LASF2245
	.byte	0x1
	.long	0xf4cf
	.long	0xf507
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x128
	.long	.LASF2246
	.byte	0x1
	.long	0xf51c
	.long	0xf55e
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x133
	.long	.LASF2247
	.byte	0x1
	.long	0xf573
	.long	0xf5c9
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1d
	.long	.LASF2063
	.byte	0xd
	.value	0x13f
	.long	.LASF2248
	.byte	0x1
	.long	0xf5de
	.long	0xf5e9
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x1428f
	.byte	0
	.uleb128 0x86
	.string	"all"
	.byte	0xd
	.value	0x144
	.long	.LASF2249
	.long	0xf2aa
	.byte	0x1
	.long	0xf605
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x95
	.long	.LASF1926
	.byte	0xd
	.value	0x14c
	.long	.LASF2250
	.long	0xf2aa
	.byte	0x1
	.uleb128 0x95
	.long	.LASF1930
	.byte	0xd
	.value	0x152
	.long	.LASF2251
	.long	0xf2aa
	.byte	0x1
	.uleb128 0x4d
	.string	"eye"
	.byte	0xd
	.value	0x158
	.long	.LASF2252
	.long	0xf2aa
	.byte	0x1
	.uleb128 0x8c
	.long	.LASF1904
	.byte	0xd
	.value	0x172
	.long	.LASF2253
	.long	0xf2aa
	.byte	0x1
	.long	0xf656
	.uleb128 0xc
	.long	0x14302
	.byte	0
	.uleb128 0x14
	.long	0xf2c6
	.uleb128 0x8c
	.long	.LASF2083
	.byte	0xd
	.value	0x17c
	.long	.LASF2254
	.long	0xf2aa
	.byte	0x1
	.long	0xf67c
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x8c
	.long	.LASF2085
	.byte	0xd
	.value	0x185
	.long	.LASF2255
	.long	0xf2aa
	.byte	0x1
	.long	0xf69d
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1f
	.string	"dot"
	.byte	0xd
	.value	0x160
	.long	.LASF2256
	.long	0x9c79
	.byte	0x1
	.long	0xf6b6
	.long	0xf6c1
	.uleb128 0xb
	.long	0x14308
	.uleb128 0xc
	.long	0x1430e
	.byte	0
	.uleb128 0x1e
	.long	.LASF1839
	.byte	0xd
	.value	0x168
	.long	.LASF2257
	.long	0x29
	.byte	0x1
	.long	0xf6da
	.long	0xf6e5
	.uleb128 0xb
	.long	0x14308
	.uleb128 0xc
	.long	0x1430e
	.byte	0
	.uleb128 0x1f
	.string	"row"
	.byte	0xd
	.value	0x1ac
	.long	.LASF2258
	.long	0xf931
	.byte	0x1
	.long	0xf6fe
	.long	0xf709
	.uleb128 0xb
	.long	0x14308
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"col"
	.byte	0xd
	.value	0x1b4
	.long	.LASF2259
	.long	0xf2aa
	.byte	0x1
	.long	0xf722
	.long	0xf72d
	.uleb128 0xb
	.long	0x14308
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1904
	.byte	0xd
	.value	0x1bf
	.long	.LASF2260
	.long	0xf2c6
	.byte	0x1
	.long	0xf746
	.long	0xf74c
	.uleb128 0xb
	.long	0x14308
	.byte	0
	.uleb128 0x1f
	.string	"t"
	.byte	0xd
	.value	0x30a
	.long	.LASF2261
	.long	0xf93b
	.byte	0x1
	.long	0xf763
	.long	0xf769
	.uleb128 0xb
	.long	0x14308
	.byte	0
	.uleb128 0x1f
	.string	"inv"
	.byte	0xd
	.value	0x34c
	.long	.LASF2262
	.long	0xf93b
	.byte	0x1
	.long	0xf782
	.long	0xf78d
	.uleb128 0xb
	.long	0x14308
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF2094
	.byte	0xd
	.value	0x3a4
	.long	.LASF2263
	.long	0xffc2
	.byte	0x1
	.long	0xf7a6
	.long	0xf7b6
	.uleb128 0xb
	.long	0x14308
	.uleb128 0xc
	.long	0x13f5f
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"mul"
	.byte	0xd
	.value	0x2c1
	.long	.LASF2264
	.long	0xf2aa
	.byte	0x1
	.long	0xf7cf
	.long	0xf7da
	.uleb128 0xb
	.long	0x14308
	.uleb128 0xc
	.long	0x1430e
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0xd
	.value	0x1c9
	.long	.LASF2265
	.long	0x14314
	.byte	0x1
	.long	0xf7f3
	.long	0xf803
	.uleb128 0xb
	.long	0x14308
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0xd
	.value	0x1d1
	.long	.LASF2266
	.long	0x142e6
	.byte	0x1
	.long	0xf81c
	.long	0xf82c
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0xd
	.value	0x1d9
	.long	.LASF2267
	.long	0x14314
	.byte	0x1
	.long	0xf845
	.long	0xf850
	.uleb128 0xb
	.long	0x14308
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0xd
	.value	0x1e1
	.long	.LASF2268
	.long	0x142e6
	.byte	0x1
	.long	0xf869
	.long	0xf874
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x1fb
	.long	.LASF2269
	.byte	0x1
	.long	0xf889
	.long	0xf89e
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x1430e
	.uleb128 0xc
	.long	0x1430e
	.uleb128 0xc
	.long	0xb48b
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x203
	.long	.LASF2270
	.byte	0x1
	.long	0xf8b3
	.long	0xf8c8
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x1430e
	.uleb128 0xc
	.long	0x1430e
	.uleb128 0xc
	.long	0xb494
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x213
	.long	.LASF2271
	.byte	0x1
	.long	0xf8dd
	.long	0xf8f2
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x1430e
	.uleb128 0xc
	.long	0x1430e
	.uleb128 0xc
	.long	0xb49d
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x229
	.long	.LASF2272
	.byte	0x1
	.long	0xf907
	.long	0xf917
	.uleb128 0xb
	.long	0x142fc
	.uleb128 0xc
	.long	0x1431a
	.uleb128 0xc
	.long	0xb4a6
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x9c79
	.uleb128 0x2e
	.string	"m"
	.long	0x30
	.byte	0x2
	.uleb128 0x2e
	.string	"n"
	.long	0x30
	.byte	0x1
	.byte	0
	.uleb128 0x2a
	.long	.LASF2273
	.uleb128 0x14
	.long	0xf2aa
	.uleb128 0x54
	.long	.LASF2274
	.byte	0x8
	.byte	0x54
	.value	0x1c3
	.long	0xffc2
	.uleb128 0x87
	.string	"val"
	.byte	0x54
	.value	0x21e
	.long	0x142ec
	.byte	0
	.byte	0x1
	.uleb128 0x42
	.long	.LASF2062
	.byte	0x54
	.value	0x1c7
	.long	0xf931
	.byte	0x1
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xce
	.long	.LASF2275
	.byte	0x1
	.long	0xf978
	.long	0xf97e
	.uleb128 0xb
	.long	0x14921
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xd3
	.long	.LASF2276
	.byte	0x1
	.long	0xf992
	.long	0xf99d
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xd9
	.long	.LASF2277
	.byte	0x1
	.long	0xf9b1
	.long	0xf9c1
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xe0
	.long	.LASF2278
	.byte	0x1
	.long	0xf9d5
	.long	0xf9ea
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xe7
	.long	.LASF2279
	.byte	0x1
	.long	0xf9fe
	.long	0xfa18
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xee
	.long	.LASF2280
	.byte	0x1
	.long	0xfa2c
	.long	0xfa4b
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xf5
	.long	.LASF2281
	.byte	0x1
	.long	0xfa5f
	.long	0xfa83
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xfe
	.long	.LASF2282
	.byte	0x1
	.long	0xfa97
	.long	0xfac0
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x107
	.long	.LASF2283
	.byte	0x1
	.long	0xfad5
	.long	0xfb03
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x110
	.long	.LASF2284
	.byte	0x1
	.long	0xfb18
	.long	0xfb4b
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x11b
	.long	.LASF2285
	.byte	0x1
	.long	0xfb60
	.long	0xfb98
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x128
	.long	.LASF2286
	.byte	0x1
	.long	0xfbad
	.long	0xfbef
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x133
	.long	.LASF2287
	.byte	0x1
	.long	0xfc04
	.long	0xfc5a
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1d
	.long	.LASF2063
	.byte	0xd
	.value	0x13f
	.long	.LASF2288
	.byte	0x1
	.long	0xfc6f
	.long	0xfc7a
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x1428f
	.byte	0
	.uleb128 0x86
	.string	"all"
	.byte	0xd
	.value	0x144
	.long	.LASF2289
	.long	0xf93b
	.byte	0x1
	.long	0xfc96
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x95
	.long	.LASF1926
	.byte	0xd
	.value	0x14c
	.long	.LASF2290
	.long	0xf93b
	.byte	0x1
	.uleb128 0x95
	.long	.LASF1930
	.byte	0xd
	.value	0x152
	.long	.LASF2291
	.long	0xf93b
	.byte	0x1
	.uleb128 0x4d
	.string	"eye"
	.byte	0xd
	.value	0x158
	.long	.LASF2292
	.long	0xf93b
	.byte	0x1
	.uleb128 0x8c
	.long	.LASF1904
	.byte	0xd
	.value	0x172
	.long	.LASF2293
	.long	0xf93b
	.byte	0x1
	.long	0xfce7
	.uleb128 0xc
	.long	0x14927
	.byte	0
	.uleb128 0x14
	.long	0xf957
	.uleb128 0x8c
	.long	.LASF2083
	.byte	0xd
	.value	0x17c
	.long	.LASF2294
	.long	0xf93b
	.byte	0x1
	.long	0xfd0d
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x8c
	.long	.LASF2085
	.byte	0xd
	.value	0x185
	.long	.LASF2295
	.long	0xf93b
	.byte	0x1
	.long	0xfd2e
	.uleb128 0xc
	.long	0x9c79
	.uleb128 0xc
	.long	0x9c79
	.byte	0
	.uleb128 0x1f
	.string	"dot"
	.byte	0xd
	.value	0x160
	.long	.LASF2296
	.long	0x9c79
	.byte	0x1
	.long	0xfd47
	.long	0xfd52
	.uleb128 0xb
	.long	0x1492d
	.uleb128 0xc
	.long	0x1431a
	.byte	0
	.uleb128 0x1e
	.long	.LASF1839
	.byte	0xd
	.value	0x168
	.long	.LASF2297
	.long	0x29
	.byte	0x1
	.long	0xfd6b
	.long	0xfd76
	.uleb128 0xb
	.long	0x1492d
	.uleb128 0xc
	.long	0x1431a
	.byte	0
	.uleb128 0x1f
	.string	"row"
	.byte	0xd
	.value	0x1ac
	.long	.LASF2298
	.long	0xf93b
	.byte	0x1
	.long	0xfd8f
	.long	0xfd9a
	.uleb128 0xb
	.long	0x1492d
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"col"
	.byte	0xd
	.value	0x1b4
	.long	.LASF2299
	.long	0xf931
	.byte	0x1
	.long	0xfdb3
	.long	0xfdbe
	.uleb128 0xb
	.long	0x1492d
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1904
	.byte	0xd
	.value	0x1bf
	.long	.LASF2300
	.long	0xf957
	.byte	0x1
	.long	0xfdd7
	.long	0xfddd
	.uleb128 0xb
	.long	0x1492d
	.byte	0
	.uleb128 0x1f
	.string	"t"
	.byte	0xd
	.value	0x30a
	.long	.LASF2301
	.long	0xf2aa
	.byte	0x1
	.long	0xfdf4
	.long	0xfdfa
	.uleb128 0xb
	.long	0x1492d
	.byte	0
	.uleb128 0x1f
	.string	"inv"
	.byte	0xd
	.value	0x34c
	.long	.LASF2302
	.long	0xf2aa
	.byte	0x1
	.long	0xfe13
	.long	0xfe1e
	.uleb128 0xb
	.long	0x1492d
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF2094
	.byte	0xd
	.value	0x3a4
	.long	.LASF2303
	.long	0xb4af
	.byte	0x1
	.long	0xfe37
	.long	0xfe47
	.uleb128 0xb
	.long	0x1492d
	.uleb128 0xc
	.long	0x14933
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"mul"
	.byte	0xd
	.value	0x2c1
	.long	.LASF2304
	.long	0xf93b
	.byte	0x1
	.long	0xfe60
	.long	0xfe6b
	.uleb128 0xb
	.long	0x1492d
	.uleb128 0xc
	.long	0x1431a
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0xd
	.value	0x1c9
	.long	.LASF2305
	.long	0x14314
	.byte	0x1
	.long	0xfe84
	.long	0xfe94
	.uleb128 0xb
	.long	0x1492d
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0xd
	.value	0x1d1
	.long	.LASF2306
	.long	0x142e6
	.byte	0x1
	.long	0xfead
	.long	0xfebd
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0xd
	.value	0x1d9
	.long	.LASF2307
	.long	0x14314
	.byte	0x1
	.long	0xfed6
	.long	0xfee1
	.uleb128 0xb
	.long	0x1492d
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0xd
	.value	0x1e1
	.long	.LASF2308
	.long	0x142e6
	.byte	0x1
	.long	0xfefa
	.long	0xff05
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x1fb
	.long	.LASF2309
	.byte	0x1
	.long	0xff1a
	.long	0xff2f
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x1431a
	.uleb128 0xc
	.long	0x1431a
	.uleb128 0xc
	.long	0xb48b
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x203
	.long	.LASF2310
	.byte	0x1
	.long	0xff44
	.long	0xff59
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x1431a
	.uleb128 0xc
	.long	0x1431a
	.uleb128 0xc
	.long	0xb494
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x213
	.long	.LASF2311
	.byte	0x1
	.long	0xff6e
	.long	0xff83
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x1431a
	.uleb128 0xc
	.long	0x1431a
	.uleb128 0xc
	.long	0xb49d
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x229
	.long	.LASF2312
	.byte	0x1
	.long	0xff98
	.long	0xffa8
	.uleb128 0xb
	.long	0x14921
	.uleb128 0xc
	.long	0x1430e
	.uleb128 0xc
	.long	0xb4a6
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x9c79
	.uleb128 0x2e
	.string	"m"
	.long	0x30
	.byte	0x1
	.uleb128 0x2e
	.string	"n"
	.long	0x30
	.byte	0x2
	.byte	0
	.uleb128 0x2a
	.long	.LASF2313
	.uleb128 0x14
	.long	0xf93b
	.uleb128 0x54
	.long	.LASF2314
	.byte	0x10
	.byte	0x54
	.value	0x1c3
	.long	0x10653
	.uleb128 0x87
	.string	"val"
	.byte	0x54
	.value	0x21e
	.long	0x1432c
	.byte	0
	.byte	0x1
	.uleb128 0x42
	.long	.LASF2062
	.byte	0x54
	.value	0x1c7
	.long	0xe225
	.byte	0x1
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xce
	.long	.LASF2315
	.byte	0x1
	.long	0x10009
	.long	0x1000f
	.uleb128 0xb
	.long	0x1433c
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xd3
	.long	.LASF2316
	.byte	0x1
	.long	0x10023
	.long	0x1002e
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xd9
	.long	.LASF2317
	.byte	0x1
	.long	0x10042
	.long	0x10052
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xe0
	.long	.LASF2318
	.byte	0x1
	.long	0x10066
	.long	0x1007b
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xe7
	.long	.LASF2319
	.byte	0x1
	.long	0x1008f
	.long	0x100a9
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xee
	.long	.LASF2320
	.byte	0x1
	.long	0x100bd
	.long	0x100dc
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xf5
	.long	.LASF2321
	.byte	0x1
	.long	0x100f0
	.long	0x10114
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x26
	.long	.LASF2063
	.byte	0xd
	.byte	0xfe
	.long	.LASF2322
	.byte	0x1
	.long	0x10128
	.long	0x10151
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x107
	.long	.LASF2323
	.byte	0x1
	.long	0x10166
	.long	0x10194
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x110
	.long	.LASF2324
	.byte	0x1
	.long	0x101a9
	.long	0x101dc
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x11b
	.long	.LASF2325
	.byte	0x1
	.long	0x101f1
	.long	0x10229
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x128
	.long	.LASF2326
	.byte	0x1
	.long	0x1023e
	.long	0x10280
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x133
	.long	.LASF2327
	.byte	0x1
	.long	0x10295
	.long	0x102eb
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1d
	.long	.LASF2063
	.byte	0xd
	.value	0x13f
	.long	.LASF2328
	.byte	0x1
	.long	0x10300
	.long	0x1030b
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0xb2ff
	.byte	0
	.uleb128 0x86
	.string	"all"
	.byte	0xd
	.value	0x144
	.long	.LASF2329
	.long	0xffcc
	.byte	0x1
	.long	0x10327
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x95
	.long	.LASF1926
	.byte	0xd
	.value	0x14c
	.long	.LASF2330
	.long	0xffcc
	.byte	0x1
	.uleb128 0x95
	.long	.LASF1930
	.byte	0xd
	.value	0x152
	.long	.LASF2331
	.long	0xffcc
	.byte	0x1
	.uleb128 0x4d
	.string	"eye"
	.byte	0xd
	.value	0x158
	.long	.LASF2332
	.long	0xffcc
	.byte	0x1
	.uleb128 0x8c
	.long	.LASF1904
	.byte	0xd
	.value	0x172
	.long	.LASF2333
	.long	0xffcc
	.byte	0x1
	.long	0x10378
	.uleb128 0xc
	.long	0x14342
	.byte	0
	.uleb128 0x14
	.long	0xffe8
	.uleb128 0x8c
	.long	.LASF2083
	.byte	0xd
	.value	0x17c
	.long	.LASF2334
	.long	0xffcc
	.byte	0x1
	.long	0x1039e
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x8c
	.long	.LASF2085
	.byte	0xd
	.value	0x185
	.long	.LASF2335
	.long	0xffcc
	.byte	0x1
	.long	0x103bf
	.uleb128 0xc
	.long	0x29
	.uleb128 0xc
	.long	0x29
	.byte	0
	.uleb128 0x1f
	.string	"dot"
	.byte	0xd
	.value	0x160
	.long	.LASF2336
	.long	0x29
	.byte	0x1
	.long	0x103d8
	.long	0x103e3
	.uleb128 0xb
	.long	0x14348
	.uleb128 0xc
	.long	0x1434e
	.byte	0
	.uleb128 0x1e
	.long	.LASF1839
	.byte	0xd
	.value	0x168
	.long	.LASF2337
	.long	0x29
	.byte	0x1
	.long	0x103fc
	.long	0x10407
	.uleb128 0xb
	.long	0x14348
	.uleb128 0xc
	.long	0x1434e
	.byte	0
	.uleb128 0x1f
	.string	"row"
	.byte	0xd
	.value	0x1ac
	.long	.LASF2338
	.long	0xe225
	.byte	0x1
	.long	0x10420
	.long	0x1042b
	.uleb128 0xb
	.long	0x14348
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"col"
	.byte	0xd
	.value	0x1b4
	.long	.LASF2339
	.long	0xffcc
	.byte	0x1
	.long	0x10444
	.long	0x1044f
	.uleb128 0xb
	.long	0x14348
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF1904
	.byte	0xd
	.value	0x1bf
	.long	.LASF2340
	.long	0xffe8
	.byte	0x1
	.long	0x10468
	.long	0x1046e
	.uleb128 0xb
	.long	0x14348
	.byte	0
	.uleb128 0x1f
	.string	"t"
	.byte	0xd
	.value	0x30a
	.long	.LASF2341
	.long	0x10658
	.byte	0x1
	.long	0x10485
	.long	0x1048b
	.uleb128 0xb
	.long	0x14348
	.byte	0
	.uleb128 0x1f
	.string	"inv"
	.byte	0xd
	.value	0x34c
	.long	.LASF2342
	.long	0x10658
	.byte	0x1
	.long	0x104a4
	.long	0x104af
	.uleb128 0xb
	.long	0x14348
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF2094
	.byte	0xd
	.value	0x3a4
	.long	.LASF2343
	.long	0xe234
	.byte	0x1
	.long	0x104c8
	.long	0x104d8
	.uleb128 0xb
	.long	0x14348
	.uleb128 0xc
	.long	0x14354
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1f
	.string	"mul"
	.byte	0xd
	.value	0x2c1
	.long	.LASF2344
	.long	0xffcc
	.byte	0x1
	.long	0x104f1
	.long	0x104fc
	.uleb128 0xb
	.long	0x14348
	.uleb128 0xc
	.long	0x1434e
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0xd
	.value	0x1c9
	.long	.LASF2345
	.long	0xb334
	.byte	0x1
	.long	0x10515
	.long	0x10525
	.uleb128 0xb
	.long	0x14348
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0xd
	.value	0x1d1
	.long	.LASF2346
	.long	0xb32e
	.byte	0x1
	.long	0x1053e
	.long	0x1054e
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x30
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0xd
	.value	0x1d9
	.long	.LASF2347
	.long	0xb334
	.byte	0x1
	.long	0x10567
	.long	0x10572
	.uleb128 0xb
	.long	0x14348
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1e
	.long	.LASF480
	.byte	0xd
	.value	0x1e1
	.long	.LASF2348
	.long	0xb32e
	.byte	0x1
	.long	0x1058b
	.long	0x10596
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x30
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x1fb
	.long	.LASF2349
	.byte	0x1
	.long	0x105ab
	.long	0x105c0
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x1434e
	.uleb128 0xc
	.long	0x1434e
	.uleb128 0xc
	.long	0xb48b
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x203
	.long	.LASF2350
	.byte	0x1
	.long	0x105d5
	.long	0x105ea
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x1434e
	.uleb128 0xc
	.long	0x1434e
	.uleb128 0xc
	.long	0xb494
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x213
	.long	.LASF2351
	.byte	0x1
	.long	0x105ff
	.long	0x10614
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x1434e
	.uleb128 0xc
	.long	0x1434e
	.uleb128 0xc
	.long	0xb49d
	.byte	0
	.uleb128 0x1c
	.long	.LASF2063
	.byte	0xd
	.value	0x229
	.long	.LASF2352
	.byte	0x1
	.long	0x10629
	.long	0x10639
	.uleb128 0xb
	.long	0x1433c
	.uleb128 0xc
	.long	0x1435a
	.uleb128 0xc
	.long	0xb4a6
	.byte	0
	.uleb128 0x2c
	.string	"_Tp"
	.long	0x29
	.uleb128 0x2e
	.string	"m"
	.long	0x30
	.byte	0x2
	.uleb128 0x2e
	.string	"n"