################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include Startup/subdir.mk
-include Src/subdir.mk
-include Drivers/STM32L0xx_HAL_Driver/Src/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 
EXECUTABLES += \
Servo\ R5.elf \

SIZE_OUTPUT += \
default.size.stdout \

OBJDUMP_LIST += \
Servo\ R5.list \

OBJCOPY_HEX += \
Servo\ R5.hex \


# All Target
all:
	+@make --no-print-directory main-build && make --no-print-directory post-build

# Main-build Target
main-build: Servo\ R5.elf secondary-outputs

# Tool invocations
Servo\ R5.elf: $(OBJS) $(USER_OBJS) C:\Users\schul\Documents\Uni\SS20\Masterarbeit\HBPMouse\HBPMouse\Servo\ R5\STM32L031G6UX_FLASH.ld
	arm-none-eabi-gcc -o "Servo R5.elf" @"objects.list" $(USER_OBJS) $(LIBS) -mcpu=cortex-m0plus -T"C:\Users\schul\Documents\Uni\SS20\Masterarbeit\HBPMouse\HBPMouse\Servo R5\STM32L031G6UX_FLASH.ld" --specs=nosys.specs -Wl,-Map="Servo R5.map" -Wl,--gc-sections -static --specs=nano.specs -mfloat-abi=soft -mthumb -Wl,--start-group -lc -lm -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

default.size.stdout: $(EXECUTABLES)
	arm-none-eabi-size  $(EXECUTABLES)
	@echo 'Finished building: $@'
	@echo ' '

Servo\ R5.list: $(EXECUTABLES)
	arm-none-eabi-objdump -h -S $(EXECUTABLES) > "Servo R5.list"
	@echo 'Finished building: $@'
	@echo ' '

Servo\ R5.hex: $(EXECUTABLES)
	arm-none-eabi-objcopy  -O ihex $(EXECUTABLES) "Servo R5.hex"
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) *
	-@echo ' '

post-build:
	rm ../Debug/Src/uart-ShowVersion.o
	-@echo ' '

secondary-outputs: $(SIZE_OUTPUT) $(OBJDUMP_LIST) $(OBJCOPY_HEX)

.PHONY: all clean dependents
.SECONDARY: post-build

-include ../makefile.targets
